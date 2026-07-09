import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.database import Base, get_db
from app.main import app

# Tests run against an in-memory SQLite DB rather than Postgres — fast,
# no external dependency, and good enough to verify API/schema behavior.
# (Note: Postgres-specific types like ARRAY are used in models.py for the
# ReID embedding column; SQLite handles this fine for structural tests
# since we don't exercise real embedding storage here.)
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def client(monkeypatch):
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db

    # Don't actually run the CV pipeline (YOLO/DeepSORT/ReID model loading)
    # during API tests — that's covered separately by pipeline-level tests
    # and isn't what these endpoint tests are checking.
    monkeypatch.setattr("app.routers.upload.process_video", lambda video_id: None)

    client = TestClient(app)
    yield client

    Base.metadata.drop_all(bind=engine)
