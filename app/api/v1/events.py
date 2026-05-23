"""Endpoint for ingesting real-time events."""
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from pydantic import BaseModel

from app.models.db import get_db
from app.models.orm import Event
from app.tasks.processing import process_event_batch

router = APIRouter()


class EventCreate(BaseModel):
    anon_id: str
    event_type: str
    details: Dict[str, Any]


@router.post("/")
def create_event(
    event_in: EventCreate,
    db: Session = Depends(get_db)
):
    """
    Accepts a single user event and saves it to the database.
    In a real system, this would write to Kafka/Redis.
    Here we save it directly to PostgreSQL and it will be picked up by Celery.
    """
    db_event = Event(
        anon_id=event_in.anon_id,
        event_type=event_in.event_type,
        details=event_in.details,
        processed=False
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    
    # Triggering Celery task immediately for the sake of real-time emulation
    # In production, this might be a cron job aggregating every X minutes
    process_event_batch.delay(event_id=db_event.id)
    
    return {"status": "ok", "event_id": db_event.id}

@router.post("/batch")
def create_events_batch(
    events_in: List[EventCreate],
    db: Session = Depends(get_db)
):
    """
    Accepts a batch of user events to simulate higher throughput.
    """
    db_events = [
        Event(
            anon_id=e.anon_id,
            event_type=e.event_type,
            details=e.details,
            processed=False
        ) for e in events_in
    ]
    db.add_all(db_events)
    db.commit()
    
    # In a real batching scenario, we might trigger a different task
    process_event_batch.delay()
    
    return {"status": "ok", "count": len(db_events)}
