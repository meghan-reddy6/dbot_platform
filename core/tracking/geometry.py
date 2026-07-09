import numpy as np
import torch
import torchvision.ops as tv_ops
from scipy.spatial.distance import cdist

def compute_ious(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes NxM IoU matrix between two sets of bounding boxes.
    Boxes should be in [x1, y1, x2, y2] format.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    
    t_boxes1 = torch.from_numpy(boxes1).float()
    t_boxes2 = torch.from_numpy(boxes2).float()
    
    iou_matrix = tv_ops.box_iou(t_boxes1, t_boxes2)
    return iou_matrix.numpy()

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
    dist_matrix = cdist(centroids_a, centroids_b, metric='euclidean')
    
    # Normalize by the width of the target boxes (boxes_b)
    widths_b = np.array([max(b[2] - b[0], 1e-6) for b in boxes_b])
    
    norm_dist_matrix = dist_matrix / widths_b
    return norm_dist_matrix

def filter_overlapping_detections(detections: list, verified_tracks: list, iou_thresh: float = 0.60) -> list:
    """
    Instantly masks out any unverified background guest detections that overlap
    significantly with an active verified track box using vectorized IoU matrix.
    """
    if len(detections) <= 1:
        return detections
        
    boxes = np.array([d["box"] for d in detections])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    iou_matrix = compute_ious(boxes, boxes)
    np.fill_diagonal(iou_matrix, 0.0)
    
    skip_indices = set()
    
    if len(verified_tracks) > 0:
        v_boxes = [p.box for p in verified_tracks]
        
        # We need distances to verified tracks
        norm_dists = compute_normalized_distances(boxes, v_boxes)
        
        # A detection maps to a verified track if normalized distance < 0.25
        maps_to_verified = np.any(norm_dists < 0.25, axis=1)
    else:
        maps_to_verified = np.zeros(len(detections), dtype=bool)
        
    for i in range(len(detections)):
        if i in skip_indices:
            continue
            
        overlaps = np.where(iou_matrix[i] > iou_thresh)[0]
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
                    
    filtered_detections = [d for idx, d in enumerate(detections) if idx not in skip_indices]
    return filtered_detections
