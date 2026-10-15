import numpy as np


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def compute_ious(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes NxM IoU matrix between two sets of bounding boxes.
    Boxes should be in [x1, y1, x2, y2] format.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    a = np.asarray(boxes1, dtype=np.float32)
    b = np.asarray(boxes2, dtype=np.float32)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.clip(bottom_right - top_left, 0.0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def compute_normalized_distances(boxes_a: list, boxes_b: list) -> np.ndarray:
    """
    Computes pairwise Euclidean distances between bounding box centroids.
    Returns distance normalized by the width of boxes_b.
    Output shape: (len(boxes_a), len(boxes_b))
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Compute centroids
    def get_centroids(boxes):
        centroids = []
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            centroids.append([cx, cy])
        return np.array(centroids)

    centroids_a = get_centroids(boxes_a)
    centroids_b = get_centroids(boxes_b)

    # Compute pairwise Euclidean distances
    dist_matrix = _pairwise_distances(centroids_a, centroids_b)

    # Normalize by the width of the target boxes (boxes_b)
    widths_b = np.array([max(b[2] - b[0], 1e-6) for b in boxes_b])

    norm_dist_matrix = dist_matrix / widths_b
    return norm_dist_matrix


def filter_overlapping_detections(
    detections: list, verified_tracks: list, iou_thresh: float = 0.60
) -> list:
    """
    Instantly masks out any unverified background guest detections that overlap
    significantly with an active verified track box using vectorized IoU and IoM matrices.
    """
    if len(detections) <= 1:
        return detections

    boxes = np.array([d["box"] for d in detections])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Compute Intersection over Min-Area (IoM) to aggressively prune "boxes within boxes"
    xA = np.maximum(boxes[:, 0][:, np.newaxis], boxes[:, 0])
    yA = np.maximum(boxes[:, 1][:, np.newaxis], boxes[:, 1])
    xB = np.minimum(boxes[:, 2][:, np.newaxis], boxes[:, 2])
    yB = np.minimum(boxes[:, 3][:, np.newaxis], boxes[:, 3])

    interArea = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    minArea = np.minimum(areas[:, np.newaxis], areas)
    
    # We combine standard IoU with IoM (Intersection over Minimum Area)
    iou_matrix = compute_ious(boxes, boxes)
    iom_matrix = interArea / (minArea + 1e-6)
    
    # Combined overlap metric catches both standard overlaps and completely enclosed boxes
    overlap_matrix = np.maximum(iou_matrix, iom_matrix)

    # Keypoint-based suppression (if two detections share the same nose, they are duplicate boxes)
    noses = np.array([d["nose"] for d in detections])
    if len(noses) > 0:
        nose_dist_matrix = _pairwise_distances(noses, noses)
        # Estimate frame width from first box (using arbitrary fallback if not available)
        box_w = boxes[0, 2] - boxes[0, 0]
        # If noses are within 10% of the box width, they are the same person
        dist_thresh = max(30, box_w * 0.1)
        overlap_matrix[nose_dist_matrix < dist_thresh] = 1.0

    np.fill_diagonal(overlap_matrix, 0.0)

    skip_indices = set()

    if len(verified_tracks) > 0:
        v_boxes = [p.box for p in verified_tracks]
        norm_dists = compute_normalized_distances(boxes, v_boxes)
        maps_to_verified = np.any(norm_dists < 0.25, axis=1)
    else:
        maps_to_verified = np.zeros(len(detections), dtype=bool)

    for i in range(len(detections)):
        if i in skip_indices:
            continue

        overlaps = np.where(overlap_matrix[i] > iou_thresh)[0]
        for j in overlaps:
            if j <= i or j in skip_indices:
                continue

            i_maps = maps_to_verified[i]
            j_maps = maps_to_verified[j]

            if i_maps and not j_maps:
                skip_indices.add(j)
            elif j_maps and not i_maps:
                skip_indices.add(i)
            else:
                if areas[i] > areas[j]:
                    skip_indices.add(j)
                else:
                    skip_indices.add(i)

    filtered_detections = [
        d for idx, d in enumerate(detections) if idx not in skip_indices
    ]
    return filtered_detections


class Keypoint:
    """
    A 2D skeletal landmark: normalized (x, y) for ratios, pixel (px, py) for angles.
    """

    def __init__(self, x_pixel: float, y_pixel: float, w: int, h: int) -> None:
        self.px: float = float(x_pixel)
        self.py: float = float(y_pixel)
        self.x: float = float(x_pixel / w)
        self.y: float = float(y_pixel / h)


def head_crop_box(pose: dict, frame_w: int, frame_h: int, min_side: float = 64.0) -> tuple:
    """Square around the head for face recognition, clipped to the frame, as int (x1, y1, x2, y2)."""
    # The detection box starts at the highest keypoint (eye level); cropping it cuts the face in half.
    nose, l_sh, r_sh = pose["nose"], pose["left_shoulder"], pose["right_shoulder"]
    shoulder_width = abs(l_sh.px - r_sh.px)
    neck_drop = abs((l_sh.py + r_sh.py) / 2.0 - nose.py)
    half = max(shoulder_width, 1.5 * neck_drop, min_side) / 2.0
    return (
        int(max(0, nose.px - half)),
        int(max(0, nose.py - half)),
        int(min(frame_w, nose.px + half)),
        int(min(frame_h, nose.py + half)),
    )
