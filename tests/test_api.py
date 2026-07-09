import io


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_camera(client):
    response = client.post("/api/cameras", json={
        "name": "Main Gate Camera",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "location_name": "Main Gate, Building A",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Main Gate Camera"
    assert "id" in data


def test_upload_video_requires_existing_camera(client):
    fake_video = io.BytesIO(b"not a real video, just testing the endpoint")
    response = client.post(
        "/api/videos/upload",
        data={"camera_id": 999, "recorded_at": "2026-01-01T10:00:00"},
        files={"file": ("test.mp4", fake_video, "video/mp4")},
    )
    assert response.status_code == 404
    assert "Camera not found" in response.json()["detail"]


def test_upload_video_success(client, tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "video_storage_path", str(tmp_path))

    camera_resp = client.post("/api/cameras", json={
        "name": "Parking Lot Camera",
        "latitude": 28.61,
        "longitude": 77.21,
    })
    camera_id = camera_resp.json()["id"]

    fake_video = io.BytesIO(b"not a real video, just testing the endpoint")
    response = client.post(
        "/api/videos/upload",
        data={"camera_id": camera_id, "recorded_at": "2026-01-01T10:00:00"},
        files={"file": ("test.mp4", fake_video, "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert "video_id" in data


def test_video_status_not_found(client):
    response = client.get("/api/videos/9999/status")
    assert response.status_code == 404


def test_search_returns_empty_list_when_no_data(client):
    response = client.post("/api/search", json={"entity_type": "person"})
    assert response.status_code == 200
    assert response.json() == []


def test_tracking_history_not_found(client):
    response = client.get("/api/entities/9999/history")
    assert response.status_code == 404


def test_map_movements_empty(client):
    response = client.get("/api/map/movements")
    assert response.status_code == 200
    assert response.json() == []
