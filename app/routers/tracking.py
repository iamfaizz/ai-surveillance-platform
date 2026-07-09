from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Entity
from app.schemas import EntityTrackHistory

router = APIRouter(prefix="/api", tags=["tracking"])


@router.get("/entities/{entity_id}/history", response_model=EntityTrackHistory)
def get_tracking_history(entity_id: int, db: Session = Depends(get_db)):
    """
    Full cross-camera movement history for one tracked entity — every
    sighting, across every camera and video, that ReID resolved to this
    same identity, ordered chronologically.
    """
    entity = db.query(Entity).filter(Entity.id == entity_id).first()
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    sightings = sorted(entity.detections, key=lambda d: d.timestamp)

    return EntityTrackHistory(
        entity_id=entity.id,
        entity_type=entity.entity_type.value,
        first_seen_at=entity.first_seen_at,
        last_seen_at=entity.last_seen_at,
        sightings=sightings,
    )
