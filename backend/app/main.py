"""TOMATO//SORT AI — FastAPI inference server.

Run from backend/:  uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import DB_PATH, UPLOAD_DIR, SettingsStore
from .db import Database
from .model_manager import ModelManager, ModelNotInstalled
from .pipeline import Pipeline
from .sources import VIDEO_EXT, SourceError, SourceSpec, SourceType, validate_video_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

models = ModelManager()
db = Database(DB_PATH)
settings = SettingsStore()
pipeline = Pipeline(models, db, settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(models.load)
    if not models.loaded:
        logging.getLogger("tomatosort").warning(models.error)
    yield
    await asyncio.to_thread(pipeline.stop)


app = FastAPI(title="TOMATO//SORT AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    type: SourceType
    value: str


# ---------------------------------------------------------------- model
@app.get("/api/health")
def health():
    return {"ok": True, "model": models.status(), "pipeline": pipeline.status()["state"]}


@app.get("/api/model")
def model_status():
    return models.status()


@app.post("/api/model/reload")
async def reload_model():
    await asyncio.to_thread(pipeline.stop)
    await asyncio.to_thread(models.load)
    return models.status()


# ---------------------------------------------------------------- sources
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXT:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Upload one of {sorted(VIDEO_EXT)}.")
    safe_name = Path(file.filename).name.replace(" ", "_")
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    with dest.open("wb") as out:
        await asyncio.to_thread(shutil.copyfileobj, file.file, out)
    try:
        info = await asyncio.to_thread(validate_video_file, dest)
    except SourceError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    return {"upload_id": dest.name, "name": file.filename, **info}


@app.post("/api/session/start")
async def start(req: StartRequest):
    if not models.loaded:
        raise HTTPException(409, models.error)
    try:
        await asyncio.to_thread(pipeline.start, SourceSpec(req.type, req.value))
    except (SourceError, ModelNotInstalled) as e:
        raise HTTPException(400, str(e))
    return pipeline.status()


@app.post("/api/session/pause")
def pause():
    pipeline.pause()
    return pipeline.status()


@app.post("/api/session/resume")
def resume():
    pipeline.resume()
    return pipeline.status()


@app.post("/api/session/stop")
async def stop():
    await asyncio.to_thread(pipeline.stop)
    return pipeline.status()


@app.post("/api/session/reset")
def reset():
    pipeline.reset()
    return pipeline.status()


@app.get("/api/status")
def status():
    return pipeline.status()


# ---------------------------------------------------------------- settings
@app.get("/api/settings")
def get_settings():
    return settings.get()


@app.patch("/api/settings")
def patch_settings(patch: dict):
    try:
        return settings.update(patch)
    except ValueError as e:
        raise HTTPException(422, str(e))


# ---------------------------------------------------------------- records
@app.get("/api/inspections")
def inspections(session_id: int | None = None, limit: int = 200):
    return db.inspections(session_id, min(max(limit, 1), 5000))


@app.get("/api/sessions")
def sessions(limit: int = 50):
    return db.sessions(min(max(limit, 1), 500))


@app.get("/api/inspections/export.csv")
def export_csv(session_id: int | None = None):
    rows = db.inspections(session_id, 1_000_000)
    buf = io.StringIO()
    fields = ["id", "session_id", "track_id", "label", "confidence", "vote_share", "observations",
              "decision", "reason", "frame_index", "created_at"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(reversed(rows))
    name = f"inspections_session_{session_id}.csv" if session_id else "inspections_all.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ---------------------------------------------------------------- live outputs
@app.get("/api/stream.mjpg")
async def stream(request: Request):
    async def frames():
        last = -1
        while not await request.is_disconnected():
            seq, jpeg = pipeline.latest_frame()
            if jpeg and seq != last:
                last = seq
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(0.015)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def telemetry(ws: WebSocket):
    await ws.accept()
    last_seq = 0
    try:
        while True:
            status_msg = pipeline.status()
            if status_msg["last_event_seq"] < last_seq:   # counters were reset
                last_seq = 0
            events = pipeline.events_since(last_seq)
            if events:
                last_seq = events[-1]["seq"]
            await ws.send_json({"type": "status", "status": status_msg, "model": {
                "installed": models.loaded, "error": models.error}, "events": events})
            await asyncio.sleep(0.25)
    except (WebSocketDisconnect, RuntimeError):
        pass
