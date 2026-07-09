from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import Base, engine
from app.routers import upload, search, tracking, map as map_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # In a real production setup you'd use Alembic migrations instead of
    # create_all — this is fine for local dev / assignment demo purposes.
    # Deliberately done at startup (not import time) so importing the app
    # module — e.g. in tests — doesn't require a live DB connection.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="AI Surveillance Platform",
    description=(
        "Multi-camera video analytics: detects and tracks people/vehicles, "
        "re-identifies entities across cameras, and exposes search + map "
        "visualization over historical footage."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(upload.router)
app.include_router(search.router)
app.include_router(tracking.router)
app.include_router(map_router.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
