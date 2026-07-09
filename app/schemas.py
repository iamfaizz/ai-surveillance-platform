from datetime import datetime
from pydantic import BaseModel, ConfigDict


class CameraCreate(BaseModel):
    name: str
    latitude: float
    longitude: float
    location_name: str | None = None


class CameraOut(CameraCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int


class VideoUploadResponse(BaseModel):
    video_id: int
    status: str
    message: str


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    video_id: int
    entity_id: int | None
    frame_number: int
    timestamp: datetime
    class_label: str
    confidence: float
    latitude: float
    longitude: float


class EntityTrackHistory(BaseModel):
    entity_id: int
    entity_type: str
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    sightings: list[DetectionOut]


class MapPoint(BaseModel):
    entity_id: int
    entity_type: str
    latitude: float
    longitude: float
    timestamp: datetime
    camera_id: int


class SearchQuery(BaseModel):
    entity_type: str | None = None
    camera_id: int | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
