"""Matching live face embeddings against registered profiles."""
import logging

import numpy as np

from .states import AMBIGUOUS_MATCH_NAME, LOW_CONFIDENCE_NAME, UNKNOWN_NAME

logger = logging.getLogger(__name__)


def _templates(profile):
    embeddings = profile.get("embeddings", [])
    if "embedding" in profile and len(embeddings) == 0:
        embeddings = [profile["embedding"]]
    return np.array(embeddings, dtype=np.float32) if len(embeddings) else None


def classify_embedding(embedding, profiles, threshold, ambiguity_margin):
    """Returns (candidate_name, best_name, similarity, margin) for a live embedding.

    candidate_name is the best profile, or an "Unknown [...]" label when the match is too weak or too
    close to a second profile to trust.
    """
    probe = embedding / (np.linalg.norm(embedding) + 1e-6)
    best, second, best_name = -1.0, -1.0, UNKNOWN_NAME
    for name, profile in profiles.items():
        templates = _templates(profile)
        if templates is None:
            continue
        if templates.shape[1] != probe.shape[0]:
            logger.warning(
                f"[BIOMETRICS] Dimension mismatch for {name}. Profile has {templates.shape[1]}-d, "
                f"live is {probe.shape[0]}-d. Skipping (please re-register)."
            )
            continue
        templates = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-6)
        similarity = np.max(templates @ probe)
        if similarity > best:
            second, best, best_name = best, similarity, name
        elif similarity > second:
            second = similarity

    several = len(profiles) > 1
    margin = best - max(second, 0.0) if several else 1.0
    if several and margin < ambiguity_margin:
        return AMBIGUOUS_MATCH_NAME, best_name, best, margin
    if best < threshold:
        return LOW_CONFIDENCE_NAME, best_name, best, margin
    return best_name, best_name, best, margin


def matching_profile(embedding, profiles, threshold):
    """Name of an existing profile this face already matches, if any."""
    probe = embedding / (np.linalg.norm(embedding) + 1e-6)
    for name, profile in profiles.items():
        templates = np.asarray(profile.get("embeddings", []), dtype=np.float32)
        if templates.ndim != 2 or templates.shape[1] != probe.shape[0]:
            continue
        templates = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-6)
        if float(np.max(templates @ probe)) >= threshold:
            return name
    return None
