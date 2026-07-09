# Architecture & Design Decisions

## Why three separate CV services instead of one pipeline function

`detection.py`, `tracking.py`, and `reid.py` are deliberately separate,
independently testable/swappable services rather than one monolithic
function, because they solve genuinely different problems:

- **Detection** (YOLOv8): "what objects are in this single frame, and
  where?" No memory of past frames.
- **Tracking** (DeepSORT): "which detections in consecutive frames of THIS
  video are the same physical object?" Gives a stable `track_id` local to
  one video.
- **Re-identification** (ReID embeddings): "is this track the same
  physical person as a track from a DIFFERENT video/camera?" This is the
  part that actually answers the assignment's core question — "track
  entities across different cameras and different points in time."

Keeping them separate means each can be improved or swapped independently
(e.g. swap YOLOv8 for a different detector) without touching the others,
and each is unit-testable in isolation.

## Why detection/tracking/ReID use pretrained models, not custom training

Given the assignment's scope (500 videos, multiple entity types, cross-
camera matching, all in a limited time window), training custom models from
scratch isn't realistic or the right use of time. The genuinely hard,
differentiated engineering problem here is building the **system** that
correctly wires detection → tracking → re-identification → storage →
queryable API together, at the intended scale. Reusing well-established
pretrained models (YOLOv8 on COCO, OSNet for person ReID) for the model
components lets the actual system architecture be the focus, which is also
what the evaluation criteria emphasize (scalability, code quality,
deployment readiness) over raw model accuracy.

## Database schema reasoning

Four tables: `Camera`, `Video`, `Entity`, `Detection`.

The key design decision is separating **Entity** from **Detection**:

- A `Detection` is a raw, low-level fact: "in frame 340 of video 7, at
  timestamp X, there's a person at this bounding box." One row per
  sighting.
- An `Entity` is a resolved, higher-level identity: "this is one specific
  person," built by ReID matching many `Detection` rows together (possibly
  from different cameras and different videos) into one identity.

This split is what makes cross-camera tracking actually queryable: `GET
/api/entities/{id}/history` can return every sighting of one person across
every camera they appeared on, ordered by time, which is the core
investigator use case the assignment describes ("track entities across
different cameras and different points in time").

Camera lat/long is denormalized onto `Detection` (copied at write time)
rather than requiring a join through `Video → Camera` on every map query,
since map retrieval is a read-heavy, latency-sensitive path.

## Why frame sampling instead of processing every frame

At 25-30fps, consecutive frames are almost identical — an object doesn't
meaningfully move in 1/25th of a second. Running full YOLO detection on
every frame would be ~25-30x more compute for negligible tracking accuracy
gain. `frame_sample_rate` (default: every 5th frame) is a tunable tradeoff
between processing cost and tracking granularity — DeepSORT's own motion
prediction handles the gaps between sampled frames reasonably well.

## Why video processing is a background task, not synchronous

A 2-hour video at even a reduced sample rate takes real time to process —
holding an HTTP request open for that long is bad API design (timeouts,
poor UX, no way to check progress). Upload returns immediately with a
`video_id`; the client polls `/api/videos/{id}/status`. This is also the
natural seam where a real deployment would swap `BackgroundTasks` for a
proper task queue (Celery/RQ) backed by Redis, which is the first thing
I'd change for production scale (see "Scaling to the full spec" below).

## Why heavy CV imports (torch, ultralytics, etc.) are deferred

`app/services/detection.py`, `tracking.py`, and `reid.py` import their
heavy dependencies (torch, ultralytics, deep-sort-realtime, torchreid)
*inside functions*, not at module top-level. This means the API layer —
routes, database models, request/response schemas — can be imported, run,
and tested without those large dependencies installed at all. It keeps the
test suite fast and makes the API contract testable independent of whether
the CV stack is even set up on a given machine (useful in CI, for
contributors working on the API-only, etc.).

## Known limitations & what "full spec" would actually require

Being upfront about the gap between this implementation and the literal
"500 videos, 2 hours each" requirement:

1. **Sequential processing.** Right now each video is processed by one
   background task on one machine. At real scale this needs a proper task
   queue with multiple GPU worker processes pulling from it in parallel.
2. **No GPU acceleration configured.** YOLOv8/DeepSORT/ReID all benefit
   massively from GPU inference. The code doesn't prevent GPU use (PyTorch
   will use CUDA if available), but nothing here provisions or requires it.
3. **Local disk storage.** `VIDEO_STORAGE_PATH` writes to local disk. At
   500 videos × up to 2 hours, this needs object storage (S3-compatible)
   plus a CDN/streaming layer rather than serving raw files off local disk.
4. **No vehicle ReID.** Vehicles are detected and tracked within a single
   video, but not matched across cameras — that needs a vehicle-specific
   ReID model (e.g. trained on VeRi-776), which wasn't in scope for this
   timeline.
5. **`create_all` instead of migrations.** Fine for a demo; a real
   deployment needs Alembic migrations for schema evolution.

None of these are hidden — the point of listing them is that the
*architecture* (schema, service boundaries, async processing model) is
designed to accommodate these upgrades without a rewrite, even though this
implementation doesn't include all of them.
