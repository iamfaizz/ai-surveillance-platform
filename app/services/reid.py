"""
Cross-camera re-identification (ReID).

This is the genuinely hard part of the assignment, and the part that's
actually novel here versus a standard "detect + track" pipeline.

The idea: a DeepSORT track_id only means something within one video. To
answer "is this the same person seen on Camera 1 at 3:00pm and Camera 2 at
3:05pm?", we need an appearance embedding — a vector that captures what a
person/vehicle LOOKS like, independent of which camera or frame it came
from. Two crops of the same person should produce similar vectors; two
different people should not.

We use a pretrained ReID model (torchreid's OSNet, trained on person-ReID
benchmarks) to generate these embeddings, then compare new detections
against known Entity embeddings using cosine similarity. If the best match
exceeds a threshold, we assign the detection to that existing Entity.
Otherwise, we create a new Entity.

Note: torchreid ships pretrained person-ReID models. Vehicle ReID would
need a separate vehicle-specific model (e.g. trained on VeRi-776) — noted
as a scope limitation in architecture.md; this implementation focuses on
person re-identification, with vehicles falling back to track-level
grouping without cross-camera matching.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_reid_model():
    """Load the pretrained OSNet ReID model once, process-wide."""
    try:
        import torchreid
    except ImportError as e:
        raise RuntimeError(
            "torchreid is not installed. Install it via requirements.txt "
            "to enable cross-camera re-identification."
        ) from e

    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=1000, pretrained=True
    )
    model.eval()
    return model


def get_embedding(person_crop) -> "np.ndarray":
    """
    Given a cropped image of a detected person (BGR numpy array from
    OpenCV), return a normalized appearance embedding vector.
    """
    import cv2
    import numpy as np
    import torch

    model = _get_reid_model()

    # Preprocessing: resize to the model's expected input, normalize.
    # (Kept minimal here — see torchreid docs for the full transform
    # pipeline used in production-grade deployments.)
    resized = cv2.resize(person_crop, (128, 256))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        embedding = model(tensor).squeeze().numpy()

    return embedding / (np.linalg.norm(embedding) + 1e-6)


def cosine_similarity(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b))


def match_or_create_entity(
    new_embedding,
    known_entities: list[tuple[int, "np.ndarray"]],
) -> tuple[int | None, float]:
    """
    Compare a new detection's embedding against all known entity embeddings.

    Returns (entity_id, similarity) of the best match if it clears the
    threshold, otherwise (None, best_similarity) meaning "create a new
    Entity" — the caller (video_processor) owns actually creating it.
    """
    from app.config import settings

    best_id, best_score = None, -1.0

    for entity_id, embedding in known_entities:
        score = cosine_similarity(new_embedding, embedding)
        if score > best_score:
            best_id, best_score = entity_id, score

    if best_score >= settings.reid_similarity_threshold:
        return best_id, best_score
    return None, best_score
