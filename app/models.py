"""
Schema design notes (also covered in architecture.md):

- Camera: a fixed physical camera with a known lat/long.
- Video: one uploaded footage file from one camera, with a real-world start
  timestamp so we can convert "frame N" into an actual wall-clock time.
- Entity: a tracked "identity" (a specific person/vehicle) that may appear
  across multiple videos/cameras. This is what cross-camera re-identification
  resolves to — many Detections can point to the same Entity.
- Detection: one bounding-box sighting of an Entity in one frame of one video.
  This is the raw signal; Entity is the resolved identity built from many
  Detections whose ReID embeddings were close enough to be considered a match.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Enum, JSON, Text
)
from sqlalchemy.orm import relationship
import enum

from app.database import Base


class VideoStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class EntityType(str, enum.Enum):
    person = "person"
    vehicle = "vehicle"
    object = "object"


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location_name = Column(String, nullable=True)

    videos = relationship("Video", back_populates="camera")


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    recorded_at = Column(DateTime, nullable=False)  # real-world start time of footage
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(Enum(VideoStatus), default=VideoStatus.pending)
    error_message = Column(Text, nullable=True)

    camera = relationship("Camera", back_populates="videos")
    detections = relationship("Detection", back_populates="video")


class Entity(Base):
    """A resolved, cross-camera tracked identity (a specific person/vehicle)."""
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True)
    entity_type = Column(Enum(EntityType), nullable=False)
    # Representative ReID embedding for this entity (e.g. running average of
    # all detection embeddings assigned to it). Used to match new detections
    # against known entities.
    reid_embedding = Column(JSON, nullable=True)
    first_seen_at = Column(DateTime, nullable=True)
    last_seen_at = Column(DateTime, nullable=True)

    detections = relationship("Detection", back_populates="entity")


class Detection(Base):
    """One bounding-box sighting of an entity in one frame of one video."""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    entity_id = Column(Integer, ForeignKey("entities.id"), nullable=True)

    frame_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False)  # recorded_at + frame offset

    class_label = Column(String, nullable=False)  # 'person', 'car', etc.
    confidence = Column(Float, nullable=False)

    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)

    # Denormalized from the camera for fast map queries without a join.
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    video = relationship("Video", back_populates="detections")
    entity = relationship("Entity", back_populates="detections")
