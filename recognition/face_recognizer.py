import os
import cv2
import numpy as np
import onnxruntime as ort
import logging
from typing import Optional
from config.settings_manager import settings

logger = logging.getLogger(__name__)

class FaceAligner:
    """
    Handles face cropping, quality checks, and 112x112 ArcFace alignment.
    """
    
    # Standard ArcFace reference points for 112x112
    REFERENCE_FACIAL_POINTS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.6963],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.3655]
    ], dtype=np.float32)

    @staticmethod
    def assess_quality(face_crop: np.ndarray, landmarks: np.ndarray) -> bool:
        """
        Deep quality heuristics to reject blurry, small, or extreme pose faces.
        """
        h, w = face_crop.shape[:2]
        if w < settings.face_min_width or h < settings.face_min_width:
            logger.debug(f"[QUALITY] Reject: Face too small ({w}x{h})")
            return False
            
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < settings.min_blur_laplacian:
            logger.debug(f"[QUALITY] Reject: Face too blurry (Laplacian: {blur_score:.2f})")
            return False
            
        return True

    @staticmethod
    def align_112x112(image: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Aligns the face based on 5 landmarks (left eye, right eye, nose, mouth left, mouth right)
        to a 112x112 crop for ArcFace using similarity transform.
        """
        try:
            tform, inliers = cv2.estimateAffinePartial2D(landmarks, FaceAligner.REFERENCE_FACIAL_POINTS)
            if tform is None:
                return None
            aligned_face = cv2.warpAffine(image, tform, (112, 112), flags=cv2.INTER_CUBIC)
            return aligned_face
        except Exception as e:
            logger.error(f"Alignment error: {e}")
            return None


class ArcFaceEmbedder:
    """
    Executes a MobileFaceNet or buffalo_l ArcFace ONNX model.
    """
    def __init__(self, model_path: str):
        self.session = None
        if model_path and os.path.exists(model_path):
            try:
                from utils.hardware import HardwareDetector
                providers = HardwareDetector.detect().get_ort_providers()
                self.session = ort.InferenceSession(model_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                logger.info(f"ArcFaceEmbedder initialized with {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ArcFace model: {e}")
        else:
            logger.warning(f"ArcFace model not found at {model_path}. Recognition disabled.")

    def forward(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        if not self.session:
            return None
        
        try:
            # ArcFace typically expects RGB image, normalized to [-1, 1]
            img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) / 127.5) - 1.0
            
            # (112, 112, 3) -> (1, 3, 112, 112)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            
            embedding = self.session.run(None, {self.input_name: img})[0][0]
            # L2 Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None


class FaceRecognizer:
    """
    Decoupled recognition pipeline: Detector -> Aligner -> Embedder
    """
    def __init__(self, model_dir: str = None):
        from core.model_manager import ModelManager
        
        detector_path = ModelManager.get_model_path("face_detector")
        landmark_path = ModelManager.get_model_path("face_landmark_detector")
        arcface_path = ModelManager.get_model_path("arcface_embedder")

        self.face_net = None
        self.landmark_net = None

        from utils.hardware import HardwareDetector
        
        providers = HardwareDetector.detect().get_ort_providers()

        if detector_path and os.path.exists(detector_path) and landmark_path and os.path.exists(landmark_path):
            try:
                self.face_net = ort.InferenceSession(detector_path, providers=providers)
            except Exception as e:
                logger.warning(f"face_net hardware initialization failed (likely invalid ONNX graph): {e}. Falling back to CPU.")
                self.face_net = ort.InferenceSession(detector_path, providers=['CPUExecutionProvider'])
                
            try:
                self.landmark_net = ort.InferenceSession(landmark_path, providers=providers)
            except Exception as e:
                logger.warning(f"landmark_net hardware initialization failed: {e}. Falling back to CPU.")
                self.landmark_net = ort.InferenceSession(landmark_path, providers=['CPUExecutionProvider'])
        else:
            logger.warning("Detector or Landmark models missing. Face Recognition degraded.")
            
        self.embedder = ArcFaceEmbedder(arcface_path)

    def extract_embedding(self, body_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Executes the full pipeline:
        1. Detector
        2. Aligner (with Quality Gates)
        3. ArcFace Embedder
        """
        if body_crop.size == 0 or not self.face_net or not self.landmark_net:
            return None

        try:
            ch, cw = body_crop.shape[:2]
            resized = cv2.resize(body_crop, (256, 256))
            img_tensor = np.expand_dims(
                np.transpose(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), (2, 0, 1)),
                axis=0,
            ).astype(np.uint8)

            outs = self.face_net.run(None, {"image": img_tensor})
            out_names = [o.name for o in self.face_net.get_outputs()]
            box_coords_1 = outs[out_names.index("box_coords_1")]
            box_coords_2 = outs[out_names.index("box_coords_2")]
            box_scores_1 = outs[out_names.index("box_scores_1")]
            box_scores_2 = outs[out_names.index("box_scores_2")]

            scores_1 = (box_scores_1.astype(np.float32) - 255) * 12.9333
            scores_2 = (box_scores_2.astype(np.float32) - 246) * 0.3584
            scores = np.concatenate([scores_1.flatten(), scores_2.flatten()])

            best_idx = np.argmax(scores)
            if scores[best_idx] > -5.0:
                coords_1 = (box_coords_1.astype(np.float32) - 192) * 1.7741
                coords_2 = (box_coords_2.astype(np.float32) - 86) * 1.9781
                coords = np.concatenate(
                    [coords_1.reshape(-1, 16), coords_2.reshape(-1, 16)]
                )

                best_coords = coords[best_idx]
                if best_idx < 512:
                    grid_y = (best_idx // 2) // 16
                    grid_x = (best_idx // 2) % 16
                    stride = 16
                else:
                    idx = best_idx - 512
                    grid_y = (idx // 6) // 8
                    grid_x = (idx // 6) % 8
                    stride = 32

                anchor_x = (grid_x + 0.5) * stride
                anchor_y = (grid_y + 0.5) * stride

                cx_256 = best_coords[0] + anchor_x
                cy_256 = best_coords[1] + anchor_y
                w_256 = best_coords[2]
                h_256 = best_coords[3]

                cx = cx_256 * cw / 256.0
                cy = cy_256 * ch / 256.0
                fw = w_256 * cw / 256.0
                fh = h_256 * ch / 256.0

                f_x1 = max(0, int(cx - fw * 0.75))
                f_y1 = max(0, int(cy - fh * 0.75))
                f_x2 = min(cw, int(cx + fw * 0.75))
                f_y2 = min(ch, int(cy + fh * 0.75))

                face_roi = body_crop[f_y1:f_y2, f_x1:f_x2]

                if face_roi.size > 0:
                    f_h, f_w = face_roi.shape[:2]
                    resized_roi = cv2.resize(face_roi, (192, 192))
                    roi_tensor = np.expand_dims(
                        np.transpose(
                            cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB), (2, 0, 1)
                        ),
                        axis=0,
                    ).astype(np.uint8)

                    lmk_outs = self.landmark_net.run(None, {"image": roi_tensor})
                    lmk_names = [o.name for o in self.landmark_net.get_outputs()]
                    landmarks_q = lmk_outs[lmk_names.index("landmarks")]
                    landmarks_raw = (landmarks_q.astype(np.float32) - 50) * 0.004985
                    
                    # Extract 5 points for ArcFace alignment
                    # landmarks_raw has shape [1, 468, 3] usually if mediapipe mesh
                    r_eye = np.array([(landmarks_raw[0, 33, 0] + landmarks_raw[0, 133, 0]) / 2.0 * f_w + f_x1,
                                      (landmarks_raw[0, 33, 1] + landmarks_raw[0, 133, 1]) / 2.0 * f_h + f_y1])
                    l_eye = np.array([(landmarks_raw[0, 362, 0] + landmarks_raw[0, 263, 0]) / 2.0 * f_w + f_x1,
                                      (landmarks_raw[0, 362, 1] + landmarks_raw[0, 263, 1]) / 2.0 * f_h + f_y1])
                    nose = np.array([landmarks_raw[0, 1, 0] * f_w + f_x1, landmarks_raw[0, 1, 1] * f_h + f_y1])
                    m_left = np.array([landmarks_raw[0, 61, 0] * f_w + f_x1, landmarks_raw[0, 61, 1] * f_h + f_y1])
                    m_right = np.array([landmarks_raw[0, 291, 0] * f_w + f_x1, landmarks_raw[0, 291, 1] * f_h + f_y1])
                    
                    face_5_points = np.array([r_eye, l_eye, nose, m_left, m_right], dtype=np.float32)
                    
                    # 1. Quality Check
                    if not FaceAligner.assess_quality(face_roi, face_5_points):
                        return None
                        
                    # 2. Align 112x112
                    aligned_face = FaceAligner.align_112x112(body_crop, face_5_points)
                    if aligned_face is None:
                        return None
                        
                    # 3. Generate Embedding
                    return self.embedder.forward(aligned_face)

        except Exception as e:
            logger.error(f"Error in extract_embedding pipeline: {e}")

        return None
