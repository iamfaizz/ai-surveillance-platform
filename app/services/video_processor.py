"""
Orchestrates the full pipeline for one uploaded video:

  frames -> detect (YOLOv8) -> track within video (DeepSORT)
         -> re-identify across cameras (ReID embeddings)
         -> persist Detections + Entities to Postgres

Runs as a FastAPI BackgroundTask so the upload endpoint can return
immediately instead of blocking for however long processing takes.
"""
from datetime import timedelta

from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import Video, VideoStatus, Detection, Entity, EntityType
from app.services.detection import detect_objects
from app.services.tracking import VideoTracker
from app.services.reid import get_embedding, match_or_create_entity


def _load_known_entity_embeddings(db: Session) -> list[tuple[int, "np.ndarray"]]:
    import numpy as np
    entities = db.query(Entity).filter(Entity.reid_embedding.isnot(None)).all()
    return [(e.id, np.array(e.reid_embedding)) for e in entities]


def process_video(video_id: int):
    """
    Entry point called as a background task after upload.
    Owns its own DB session since it runs outside the request lifecycle.
    """
    import cv2

    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if video is None:
            return

        video.status = VideoStatus.processing
        db.commit()

        cap = cv2.VideoCapture(video.storage_path)
        if not cap.isOpened():
            video.status = VideoStatus.failed
            video.error_message = "Could not open video file"
            db.commit()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        tracker = VideoTracker()
        known_embeddings = _load_known_entity_embeddings(db)

        frame_number = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Skip frames per the configured sample rate — running full
            # detection on every single frame is unnecessary and slow;
            # objects don't meaningfully move frame-to-frame at 25-30fps.
            if frame_number % settings.frame_sample_rate != 0:
                frame_number += 1
                continue

            timestamp = video.recorded_at + timedelta(seconds=frame_number / fps)

            detections = detect_objects(frame)
            tracks = tracker.update(frame, detections)

            for track in tracks:
                x1, y1, x2, y2 = [int(v) for v in track.bbox]
                x1, y1 = max(x1, 0), max(y1, 0)
                crop = frame[y1:y2, x1:x2]

                entity_id = None
                if track.class_label == "person" and crop.size > 0:
                    embedding = get_embedding(crop)
                    matched_id, _ = match_or_create_entity(embedding, known_embeddings)

                    if matched_id is not None:
                        entity_id = matched_id
                    else:
                        new_entity = Entity(
                            entity_type=EntityType.person,
                            reid_embedding=embedding.tolist(),
                            first_seen_at=timestamp,
                            last_seen_at=timestamp,
                        )
                        db.add(new_entity)
                        db.flush()  # get the new entity's id without committing
                        entity_id = new_entity.id
                        known_embeddings.append((entity_id, embedding))
                else:
                    # Vehicles: no cross-camera ReID model wired up yet
                    # (see reid.py note) — grouped per-track only for now.
                    new_entity = Entity(
                        entity_type=EntityType.vehicle
                        if track.class_label in ("car", "truck", "motorcycle", "bus")
                        else EntityType.object,
                        first_seen_at=timestamp,
                        last_seen_at=timestamp,
                    )
                    db.add(new_entity)
                    db.flush()
                    entity_id = new_entity.id

                db.add(Detection(
                    video_id=video.id,
                    entity_id=entity_id,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    class_label=track.class_label,
                    confidence=track.confidence,
                    bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                    latitude=video.camera.latitude,
                    longitude=video.camera.longitude,
                ))

            frame_number += 1

        cap.release()
        video.status = VideoStatus.done
        db.commit()

    except Exception as e:  # noqa: BLE001 - top-level pipeline guard, logged & persisted
        db.rollback()
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = VideoStatus.failed
            video.error_message = str(e)
            db.commit()
    finally:
        db.close()
