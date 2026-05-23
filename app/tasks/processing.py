import logging
from sqlalchemy.orm import Session
from app.models.db import SessionLocal
from app.models.orm import Event

logger = logging.getLogger(__name__)

try:
    from celery import shared_task
except ModuleNotFoundError:
    class _EagerTask:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def delay(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    def shared_task(fn):
        return _EagerTask(fn)


@shared_task
def process_event_batch(event_id: int = None):
    """
    Process new events. If event_id is provided, process that specific event.
    Otherwise, process a batch of unprocessed events.
    """
    db = SessionLocal()
    try:
        if event_id:
            event = db.query(Event).filter(Event.id == event_id).first()
            if event and not event.processed:
                # Placeholder for real aggregation logic
                # Here we would update the user's daily aggregates in Redis or Postgres
                # and if a full day has passed, we trigger the ML model for anomaly detection.
                event.processed = True
                db.commit()
                logger.info(f"Processed event {event_id}")
        else:
            events = db.query(Event).filter(Event.processed == False).limit(100).all()
            for event in events:
                event.processed = True
            db.commit()
            logger.info(f"Processed batch of {len(events)} events")
    except Exception as e:
        db.rollback()
        logger.error(f"Error processing events: {e}")
    finally:
        db.close()
