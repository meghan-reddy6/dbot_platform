import os
import cv2
import numpy as np
import logging
from typing import Optional
import onnxruntime as ort
from utils.hardware import HardwareDetector

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Executes Biometric Face Recognition.
    Wraps the ONNX Face Detector, Landmark Aligner, and LBP Cascade
    into a unified pipeline for the multiprocessing worker.
    """

    def __init__(self, model_dir: str = "D:\\Thundersoft\\dbot\\models"):
        self.model_dir = model_dir
        HardwareDetector.detect()

        # In a real edge system, we would inject QNN or CoreML ExecutionProviders here.
        # For compatibility with the old script, we fallback to CPU.
        providers = ["CPUExecutionProvider"]

        model_path = os.path.join(model_dir, "face_detector.onnx")
        landmark_path = os.path.join(model_dir, "face_landmark_detector.onnx")

        try:
            self.face_net = ort.InferenceSession(model_path, providers=providers)
            self.landmark_net = ort.InferenceSession(landmark_path, providers=providers)
            self.stage2_cascade = NativeFaceCascade(model_dir)
            logger.info("FaceRecognizer backend initialized successfully.")
        except Exception as e:
            logger.error(f"FaceRecognizer ONNX load failed: {e}")
            self.face_net = None
            self.landmark_net = None
            self.stage2_cascade = None

    def extract_embedding(self, body_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Takes a full body crop and returns a 128-dimensional embedding.
        """
        if body_crop.size == 0 or not self.face_net:
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

                det_info = {}
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
                    landmarks = (landmarks_q.astype(np.float32) - 50) * 0.004985

                    r_eye_x = (
                        landmarks[0, 33, 0] + landmarks[0, 133, 0]
                    ) / 2.0 * f_w + f_x1
                    r_eye_y = (
                        landmarks[0, 33, 1] + landmarks[0, 133, 1]
                    ) / 2.0 * f_h + f_y1
                    l_eye_x = (
                        landmarks[0, 362, 0] + landmarks[0, 263, 0]
                    ) / 2.0 * f_w + f_x1
                    l_eye_y = (
                        landmarks[0, 362, 1] + landmarks[0, 263, 1]
                    ) / 2.0 * f_h + f_y1

                    dY = r_eye_y - l_eye_y
                    dX = r_eye_x - l_eye_x
                    angle = np.degrees(np.arctan2(dY, dX)) - 180

                    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                    aligned_crop = cv2.warpAffine(
                        body_crop, M, (cw, ch), flags=cv2.INTER_CUBIC
                    )

                    size = int(max(fw, fh) * 1.2)
                    half_size = size // 2

                    crop_y1, crop_y2 = (
                        max(0, int(cy - half_size)),
                        min(ch, int(cy + half_size)),
                    )
                    crop_x1, crop_x2 = (
                        max(0, int(cx - half_size)),
                        min(cw, int(cx + half_size)),
                    )

                    if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                        det_info["roi_frame"] = aligned_crop[
                            crop_y1:crop_y2, crop_x1:crop_x2
                        ]

                is_pre_aligned = "roi_frame" in det_info

                final_crop = det_info.get("roi_frame", body_crop)
                if self.stage2_cascade:
                    return self.stage2_cascade.generate_signature(
                        final_crop, is_pre_aligned
                    )
                else:
                    return np.zeros(128, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error in extract_embedding: {e}")

        return None


class NativeFaceCascade:
    """
    Independent Stage 2: Native Biometric Facial Execution Cascade
    Executes OpenCV HAAR Face Detection -> LBP Texture Extraction
    """

    def __init__(self, model_dir: str):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def generate_signature(
        self, face_crop: np.ndarray, pre_aligned: bool = False
    ) -> np.ndarray:
        """
        Executes cascade and produces a 128-d LBP texture signature.
        Divides the face into an 8x8 grid (64 cells). Each cell generates a 2-bin LBP histogram.
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        if not pre_aligned:
            # 1. Detection
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )

            if len(faces) == 0:
                return np.zeros(128, dtype=np.float32)

            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            inner_face = gray[y : y + h, x : x + w]
        else:
            inner_face = gray

        # 2. Extract LBP 128-d
        inner_face = cv2.resize(inner_face, (64, 64))
        lbp = np.zeros_like(inner_face)
        for i in range(1, 63):
            for j in range(1, 63):
                center = inner_face[i, j]
                code = 0
                code |= (inner_face[i - 1, j - 1] > center) << 7
                code |= (inner_face[i - 1, j] > center) << 6
                code |= (inner_face[i - 1, j + 1] > center) << 5
                code |= (inner_face[i, j + 1] > center) << 4
                code |= (inner_face[i + 1, j + 1] > center) << 3
                code |= (inner_face[i + 1, j] > center) << 2
                code |= (inner_face[i + 1, j - 1] > center) << 1
                code |= (inner_face[i, j - 1] > center) << 0
                lbp[i, j] = code

        features = []
        cell_size = 8
        for i in range(8):
            for j in range(8):
                cell = lbp[
                    i * cell_size : (i + 1) * cell_size,
                    j * cell_size : (j + 1) * cell_size,
                ]
                hist = cv2.calcHist([cell], [0], None, [2], [0, 256])
                features.extend(hist.flatten())

        embedding = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding
