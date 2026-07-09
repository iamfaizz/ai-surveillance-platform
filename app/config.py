"""
Central configuration, loaded from environment variables (.env file locally,
real env vars in Docker/production).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://surveillance:surveillance@db:5432/surveillance"

    # Storage
    video_storage_path: str = "/data/videos"
    max_video_duration_minutes: int = 120  # 2 hours, per spec
    max_videos_supported: int = 500

    # Detection / tracking
    yolo_model_path: str = "yolov8n.pt"  # nano model: fast, good enough for a demo
    yolo_confidence_threshold: float = 0.4
    reid_similarity_threshold: float = 0.7  # cosine similarity cutoff for "same entity"
    detection_classes: list[str] = ["person", "car", "truck", "motorcycle", "bus"]

    # Frame sampling — we don't run detection on every single frame,
    # both for speed and because objects don't move much frame-to-frame.
    frame_sample_rate: int = 5  # process 1 out of every N frames

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
