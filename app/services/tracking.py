"""
Within-camera tracking using DeepSORT.

Detection alone tells you "there's a person here in this frame." Tracking
tells you "this is the same person as 10 frames ago" — it assigns a stable
track_id to an object as it moves across frames in a SINGLE video/camera.

This is deliberately a separate concern from cross-camera re-identification
(see reid.py). DeepSORT tracks are local to one video; ReID is what stitches
tracks from DIFFERENT cameras into one global Entity.
"""
from dataclasses import dataclass

from app.services.detection import Detection


@dataclass
class Track:
    track_id: int
    class_label: str
    bbox: tuple[float, float, float, float]
    confidence: float


class VideoTracker:
    """
    One instance per video being processed — DeepSORT keeps internal state
    (track history) that must not be shared across different videos.
    """

    def __init__(self):
        # Deferred import — see detection.py's _get_model for why: keeps
        # the API layer runnable/testable without the heavy CV deps
        # installed. Only actual video processing needs this.
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._tracker = DeepSort(max_age=30)

    def update(self, frame, detections: list[Detection]) -> list[Track]:
        # deep-sort-realtime expects [[x1, y1, w, h], confidence, class_name]
        formatted = [
            ([d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]],
             d.confidence, d.class_label)
            for d in detections
        ]

        tracks = self._tracker.update_tracks(formatted, frame=frame)

        results: list[Track] = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_ltrb()
            results.append(
                Track(
                    track_id=t.track_id,
                    class_label=t.det_class or "unknown",
                    bbox=(x1, y1, x2, y2),
                    confidence=t.det_conf or 0.0,
                )
            )
        return results
