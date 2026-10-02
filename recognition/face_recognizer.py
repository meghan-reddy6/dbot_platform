import json
import os
import cv2
import numpy as np
import logging
from typing import Optional
from config.settings_manager import settings

logger = logging.getLogger(__name__)

# (scale, zero_point) of the uint8 outputs of the AI Hub w8a8 MediaPipe exports. models/metadata.json
# ships with the models and wins, so a re-exported model does not decode with stale numbers.
_DEFAULT_QUANTIZATION = {
    "box_coords_1": (1.7741001844406128, 192),
    "box_coords_2": (1.9781091213226318, 86),
    "box_scores_1": (12.93332576751709, 255),
    "box_scores_2": (0.3584882616996765, 246),
    "landmarks": (0.004985911305993795, 50),
    "scores": (0.00390625, 0),
}
# The landmark model's face-presence score. Measured on sample images: 0.996 on real faces and
# exactly 0.500 on every non-face region, which the pipeline used to embed anyway.
MIN_FACE_PRESENCE = 0.75


def load_quantization(metadata_path: str) -> dict:
    params = dict(_DEFAULT_QUANTIZATION)
    try:
        with open(metadata_path, encoding="utf-8") as f:
            model_files = json.load(f).get("model_files", {})
        for model in ("face_detector.onnx", "face_landmark_detector.onnx"):
            for name, spec in model_files.get(model, {}).get("outputs", {}).items():
                quant = spec.get("quantization_parameters")
                if quant:
                    params[name] = (float(quant["scale"]), int(quant["zero_point"]))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        logger.warning(f"Using built-in face model quantization parameters ({metadata_path}: {e})")
    return params


# The face models run in the worker process next to the pose model; a few threads keep them out of its way.
_FACE_MODEL_THREADS = 2


def _create_session(path: str, providers: list):
    from utils.hardware import arm64_cpu_needs_unfused_quantized, create_session

    try:
        session = create_session(path, providers, threads=_FACE_MODEL_THREADS)
    except Exception as e:
        if providers == ["CPUExecutionProvider"]:
            raise
        logger.warning(f"{os.path.basename(path)} failed on {providers}: {e}. Falling back to CPU.")
        return create_session(path, ["CPUExecutionProvider"], threads=_FACE_MODEL_THREADS)
    if arm64_cpu_needs_unfused_quantized() and session.get_providers()[0] != "CPUExecutionProvider":
        # ONNX Runtime's own fallback would rebuild the session on the CPU without create_session's ARM setting;
        # an accelerator failure instead reaches the worker, which rebuilds the recognizer through this function.
        session.disable_fallback()
    return session

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
                self.session = _create_session(model_path, providers)
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
        # Read by the biometric worker, which rebuilds the recognizer after repeated failures.
        self.consecutive_errors = 0
        self.quantization = load_quantization(
            os.path.join(model_dir or ModelManager._models_dir, "metadata.json")
        )

        from utils.hardware import HardwareDetector
        
        providers = HardwareDetector.detect().get_ort_providers()

        if detector_path and os.path.exists(detector_path) and landmark_path and os.path.exists(landmark_path):
            # DirectML rejects this quantized detector ("two nodes with same node name"); it takes ~2 ms on the CPU.
            detector_providers = [p for p in providers if p != "DmlExecutionProvider"] or ["CPUExecutionProvider"]
            self.face_net = _create_session(detector_path, detector_providers)
            self.landmark_net = _create_session(landmark_path, providers)
        else:
            logger.warning("Detector or Landmark models missing. Face Recognition degraded.")
            
        self.embedder = ArcFaceEmbedder(arcface_path)

    def _dequantize(self, name: str, values: np.ndarray) -> np.ndarray:
        scale, zero_point = self.quantization[name]
        return (values.astype(np.float32) - zero_point) * scale

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
            embedding = self._extract(body_crop)
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Error in extract_embedding pipeline: {e}")
            return None
        self.consecutive_errors = 0
        return embedding

    def _extract(self, body_crop: np.ndarray) -> Optional[np.ndarray]:
        ch, cw = body_crop.shape[:2]
        # Resize keeping the aspect ratio and pad to 256x256, as MediaPipe expects; head crops
        # clipped at the frame edge are not square and a plain resize would squash the face.
        scale = 256.0 / max(ch, cw)
        nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
        pad_x, pad_y = (256 - nw) // 2, (256 - nh) // 2
        resized = np.zeros((256, 256, 3), np.uint8)
        resized[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(body_crop, (nw, nh))
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

        scores_1 = self._dequantize("box_scores_1", box_scores_1)
        scores_2 = self._dequantize("box_scores_2", box_scores_2)
        scores = np.concatenate([scores_1.flatten(), scores_2.flatten()])

        best_idx = np.argmax(scores)
        if scores[best_idx] > -5.0:
            coords_1 = self._dequantize("box_coords_1", box_coords_1)
            coords_2 = self._dequantize("box_coords_2", box_coords_2)
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

            cx = (cx_256 - pad_x) / scale
            cy = (cy_256 - pad_y) / scale
            fw = w_256 / scale
            fh = h_256 / scale

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
                if "scores" in lmk_names:
                    presence = float(self._dequantize("scores", lmk_outs[lmk_names.index("scores")]).ravel()[0])
                    if presence < MIN_FACE_PRESENCE:
                        logger.debug(f"[QUALITY] Reject: no face in detector box (presence {presence:.2f})")
                        return None
                landmarks_q = lmk_outs[lmk_names.index("landmarks")]
                landmarks_raw = self._dequantize("landmarks", landmarks_q)
                
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

        return None
