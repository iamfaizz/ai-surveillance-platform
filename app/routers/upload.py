import os
import shutil
from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Video, VideoStatus, Camera
from app.schemas import VideoUploadResponse, CameraCreate, CameraOut
from app.services.video_processor import process_video

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/cameras", response_model=CameraOut)
def register_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    """Register a physical camera and its fixed lat/long before uploading its footage."""
    db_camera = Camera(**camera.model_dump())
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    return db_camera


@router.post("/videos/upload", response_model=VideoUploadResponse)
def upload_video(
    background_tasks: BackgroundTasks,
    camera_id: int = Form(...),
    recorded_at: datetime = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Accepts one video file for one camera. Processing (detection, tracking,
    ReID) happens asynchronously in the background — this endpoint returns
    immediately with a video_id you can poll via GET /api/videos/{id}/status.
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found. Register it first via POST /api/cameras.")

    os.makedirs(settings.video_storage_path, exist_ok=True)
    storage_path = os.path.join(settings.video_storage_path, f"{camera_id}_{file.filename}")

    with open(storage_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video = Video(
        camera_id=camera_id,
        filename=file.filename,
        storage_path=storage_path,
        recorded_at=recorded_at,
        status=VideoStatus.pending,
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    background_tasks.add_task(process_video, video.id)

    return VideoUploadResponse(
        video_id=video.id,
        status=video.status.value,
        message="Video accepted. Processing started in the background.",
    )


@router.get("/videos/{video_id}/status")
def get_video_status(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return {
        "video_id": video.id,
        "status": video.status.value,
        "error_message": video.error_message,
    }
