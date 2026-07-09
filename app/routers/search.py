from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Detection
from app.schemas import DetectionOut, SearchQuery

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=list[DetectionOut])
def search_detections(query: SearchQuery, db: Session = Depends(get_db)):
    """
    Search historical footage by entity type, camera, and/or time window.
    This is the primary tool investigators use to answer
    "was a vehicle/person seen at Camera X between time A and time B?"
    """
    q = db.query(Detection)

    if query.entity_type:
        q = q.filter(Detection.class_label == query.entity_type)
    if query.camera_id:
        q = q.join(Detection.video).filter(Detection.video.has(camera_id=query.camera_id))
    if query.start_time:
        q = q.filter(Detection.timestamp >= query.start_time)
    if query.end_time:
        q = q.filter(Detection.timestamp <= query.end_time)

    return q.order_by(Detection.timestamp.desc()).limit(500).all()
