from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Detection
from app.schemas import MapPoint

router = APIRouter(prefix="/api", tags=["map"])


@router.get("/map/movements", response_model=list[MapPoint])
def get_movement_points(
    entity_id: int | None = Query(None),
    start_time: datetime | None = Query(None),
    end_time: datetime | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Returns timestamped lat/long points, ordered chronologically, that the
    frontend map draws as a path — one entity's movement across cameras,
    or (if entity_id is omitted) all recent activity for a general
    overview map.
    """
    q = db.query(Detection)

    if entity_id:
        q = q.filter(Detection.entity_id == entity_id)
    if start_time:
        q = q.filter(Detection.timestamp >= start_time)
    if end_time:
        q = q.filter(Detection.timestamp <= end_time)

    detections = q.order_by(Detection.timestamp.asc()).limit(1000).all()

    return [
        MapPoint(
            entity_id=d.entity_id,
            entity_type=d.entity.entity_type.value if d.entity else "unknown",
            latitude=d.latitude,
            longitude=d.longitude,
            timestamp=d.timestamp,
            camera_id=d.video.camera_id,
        )
        for d in detections
        if d.entity_id is not None
    ]
