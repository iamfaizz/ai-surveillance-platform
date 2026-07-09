"""
Object detection using a pretrained YOLOv8 model (via Ultralytics).

We don't train our own detector — YOLOv8's COCO-pretrained weights already
cover person/car/truck/motorcycle/bus, which is all this platform needs.
This keeps the project honest about what's genuinely novel here (the
tracking + cross-camera re-ID + the system built around it) versus what's
a solved problem we're reusing (raw object detection).
"""
from dataclasses import dataclass
from functools import lru_cache

from app.config import settings

# COCO class names we care about, mapped to our own simplified labels.
COCO_CLASS_MAP = {
    "person": "person",
    "car": "car",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "bus": "bus",
}


@dataclass
class Detection:
    class_label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


@lru_cache(maxsize=1)
def _get_model():
    """
    Load the model once and cache it — loading YOLO weights per-frame would
    be disastrously slow. lru_cache gives us a process-wide singleton.

    Import is deliberately deferred to here (not module top-level) so the
    rest of the API — routes, schemas, DB models, tests — can run without
    requiring the ultralytics/torch install. Only actual video processing
    needs it.
    """
    from ultralytics import YOLO
    return YOLO(settings.yolo_model_path)


def detect_objects(frame) -> list[Detection]:
    """
    Run detection on a single video frame (a numpy array from OpenCV).
    Returns only the classes we care about, above the confidence threshold.
    """
    model = _get_model()
    results = model(frame, verbose=False)[0]

    detections: list[Detection] = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name not in COCO_CLASS_MAP:
            continue

        confidence = float(box.conf[0])
        if confidence < settings.yolo_confidence_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            Detection(
                class_label=COCO_CLASS_MAP[class_name],
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
            )
        )

    return detections
