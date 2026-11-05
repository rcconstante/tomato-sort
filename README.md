# TOMATO//SORT AI

Real-time tomato ripeness grading and conveyor-belt inspection.

A YOLO11 detector trained on the Roboflow dataset
[`ylli-uzfdv/tomato-ripeness-tb3ho`](https://universe.roboflow.com/ylli-uzfdv/tomato-ripeness-tb3ho)
finds every tomato in the frame. ByteTrack follows each one, and the tomato is graded **once**, when it crosses a
virtual inspection line. A simulated sorter then routes it to **ACCEPT**, **REJECT** or **REVIEW**.

```
notebooks/TomatoSort_AI_Training.ipynb   Colab notebook: download → analyse → train → evaluate → export
backend/                                 FastAPI + Ultralytics YOLO + ByteTrack + OpenCV + SQLite
frontend/                                React + Vite inspection dashboard
tools/build_notebook.py                  regenerates the notebook from readable Python source
```

## 1. Train the model (Google Colab)

1. Open `notebooks/TomatoSort_AI_Training.ipynb` in Colab (File → Upload notebook).
2. Runtime → Change runtime type → **T4 GPU**.
3. Add your Roboflow API key as a Colab secret named `ROBOFLOW_API_KEY` (🔑 sidebar). Without a secret, the notebook asks for it in a hidden prompt.
4. Run all cells. The notebook:
   - lists the published dataset versions and uses the latest one (or the one you set in `DATASET_VERSION`),
   - reads the class names from `data.yaml`, checks that every image has a label file and shows annotated samples,
   - trains `yolo11n.pt` for up to 80 epochs (patience 15) with hue augmentation turned off,
   - evaluates on the validation split, and on the test split if the dataset has one,
   - exports `best.pt`, `last.pt`, `model.onnx`, `model_metadata.json` and `training_metrics.json`.
5. Copy `best.pt` and `model_metadata.json` into `backend/models/`.

If dataset setup reports a missing validation split, check the directory listing in the error. The notebook
recognizes `val`, `valid`, and `validation` folders, including exports with missing or stale YAML paths.
It requires separate, non-empty training and validation image directories. When the export contains
only training images, it creates a local, seeded 80/20 holdout by source-filename group, copying images
and labels into separate folders while preserving the original download. Roboflow variants sharing
the filename before `.rf.` stay together. This holdout is not an official dataset validation split and
does not guarantee separation of related video frames or renamed duplicates. For final evaluation,
prefer a version split by original capture/source before augmentation. Images without label files
must be checked and labeled before the holdout can be created (empty files are valid for confirmed backgrounds).
Do not point validation at the training images. After updating this notebook, upload the new copy to
Colab and rerun the download and dataset-validation cells before training.

The model only knows the classes in the dataset (`ripe` and `unripe`). Rotten, damaged and semi-ripe tomatoes are
outside its scope.

## 2. Run the inspection server

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For an NVIDIA GPU, install the CUDA build of PyTorch first (see pytorch.org). The server uses `cuda:0` automatically
when it is available.

Without `backend/models/best.pt` the server still starts, but it reports *model not installed* and refuses to run
inspection. It never substitutes another model or fake detections.

## 3. Run the dashboard

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server proxies `/api` and `/ws` to `127.0.0.1:8000`.

## How it works

| Stage | Detail |
|---|---|
| Sources | Uploaded MP4, local video file path, webcam (index), RTSP or HTTP stream. Start / pause / resume / stop / reset. Live sources reconnect up to 5 times with backoff. |
| Detection | `model.track(..., tracker="bytetrack.yaml", persist=True)` on every frame. |
| Colours | Ripe = green, unripe = orange, below the *certain* threshold = grey. Each box shows `#track label confidence` and, once graded, its decision. |
| Counting | Each track's centre is tested against the inspection line. A tomato is counted only when it crosses from one side to the other in the chosen direction, within the segment's extent, and only once per track. |
| Grading | Every frame of a track adds a confidence-weighted vote (frames below the *certain* threshold count ¼). At the crossing, the track becomes **uncertain** instead of being forced into a class if it has fewer than *N* frames, its class votes disagree, or its mean confidence is low. |
| Sorting | Default rules: ripe → ACCEPT, unripe → REJECT, uncertain → REVIEW. Rules can be changed per class in the dashboard. |
| State resets | Each new source gets a fresh model instance, so ByteTrack IDs never carry over. Looping a video file or pressing Reset also clears tracking state. |
| Storage | `backend/data/inspections.db` (SQLite) stores sessions and every inspection event. Export as CSV from the dashboard or `GET /api/inspections/export.csv`. |

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/model` | Server and model status (classes, device, metadata) |
| POST | `/api/model/reload` | Reload `best.pt` after copying it in |
| POST | `/api/upload` | Upload a video (multipart `file`) and get back an `upload_id` |
| POST | `/api/session/start` | `{"type": "upload" \| "file" \| "webcam" \| "rtsp", "value": "..."}` |
| POST | `/api/session/pause` · `resume` · `stop` · `reset` | Session control |
| GET / PATCH | `/api/settings` | Thresholds, line position and direction, display toggles, sorting rules |
| GET | `/api/stream.mjpg` | Annotated MJPEG stream |
| WS | `/ws` | Status every 250 ms and new inspection events (with a crop image of each tomato) |
| GET | `/api/inspections`, `/api/sessions`, `/api/inspections/export.csv` | Stored records |

## Tests

```bash
cd backend
pip install pytest
python -m pytest -q
```

The tests cover vote finalisation, line-crossing direction and count-once behaviour.
