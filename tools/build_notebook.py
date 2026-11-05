"""Builds notebooks/TomatoSort_AI_Training.ipynb.

The notebook is generated from this file so the cell sources stay readable
and diffable. Run:  python tools/build_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "TomatoSort_AI_Training.ipynb"

cells = []


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip("\n"),
    })


# ---------------------------------------------------------------- Cell 1
md(r"""
# TOMATO//SORT AI
## Automated Tomato Grading Using YOLO

**Objective.** Train an object detector that finds every tomato on a production conveyor belt and classifies it as
`ripe` or `unripe`. The trained weights power the local TOMATO//SORT AI inspection app (FastAPI + ByteTrack + React),
which counts each tomato once at a virtual inspection line and simulates ACCEPT / REJECT / REVIEW sorting.

**Dataset.** Roboflow Universe — workspace `ylli-uzfdv`, project `tomato-ripeness-tb3ho`
(object detection, ~2,874 images across published versions). The class list is read from the downloaded
`data.yaml`; nothing in this notebook assumes a class index.

**Model.** Ultralytics YOLO11n (`yolo11n.pt`, COCO-pretrained) fine-tuned with transfer learning.
Switch to YOLO11s in the configuration cell for higher accuracy if your GPU allows.

**Scope.** The model only knows the classes in the dataset (expected: `ripe`, `unripe`).
Rotten, moldy, damaged and semi-ripe tomatoes are **not** supported classes in this first model.

**Output files** (in `/content/tomatosort_export/`):

| File | Purpose |
|---|---|
| `best.pt` | Best checkpoint (by validation fitness) → copy to `backend/models/best.pt` |
| `last.pt` | Last checkpoint, for resuming |
| `model.onnx` | Optional ONNX export |
| `model_metadata.json` | Architecture, class map, dataset version, image size, training date, measured metrics |
| `training_metrics.json` | Per-epoch metrics from `results.csv` plus final validation/test metrics |

**Runtime.** `Runtime → Change runtime type → T4 GPU` is recommended. CPU works but training is very slow.
Run the cells top to bottom.
""")

# ---------------------------------------------------------------- Cell 2
md("## 1 · Environment setup")
code(r"""
%pip install -q "ultralytics>=8.3.0" "roboflow>=1.1.40" opencv-python-headless matplotlib pandas pyyaml seaborn onnx onnxslim

import sys, platform, torch, ultralytics

print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
print(f"PyTorch     : {torch.__version__}")
print(f"Ultralytics : {ultralytics.__version__}")
CUDA = torch.cuda.is_available()
print(f"CUDA        : {CUDA}")
if CUDA:
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU         : {props.name}")
    print(f"GPU memory  : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    DEVICE = 0
else:
    DEVICE = "cpu"
    print("\n⚠️  No GPU detected. Training will run on CPU and can take many hours.")
    print("   In Colab choose Runtime → Change runtime type → T4 GPU, then re-run from the top.")

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
print(f"Running in Colab: {IN_COLAB}")
""")

# ---------------------------------------------------------------- Cell 3
md(r"""
## 2 · Download the dataset from Roboflow

The API key is read from **Colab Secrets** (🔑 icon in the left sidebar → add `ROBOFLOW_API_KEY` and enable notebook access).
If no secret is set you are prompted with a hidden `getpass` input. The key is never printed or saved.

Set `DATASET_VERSION = None` to use the latest published version, or pin a number for reproducibility.
""")
code(r"""
import os, re, getpass, random, yaml
from pathlib import Path
from roboflow import Roboflow

WORKSPACE = "ylli-uzfdv"
PROJECT = "tomato-ripeness-tb3ho"
DATASET_VERSION = None   # None = latest published version, or e.g. 3
EXPORT_FORMAT = "yolov8"

def get_api_key():
    try:
        from google.colab import userdata
        key = userdata.get("ROBOFLOW_API_KEY")
        if key:
            print("Using ROBOFLOW_API_KEY from Colab Secrets.")
            return key
    except Exception:
        pass
    return getpass.getpass("Roboflow API key (hidden): ").strip()

rf = Roboflow(api_key=get_api_key())
project = rf.workspace(WORKSPACE).project(PROJECT)

def version_number(v):
    raw = getattr(v, "version", None) or getattr(v, "id", "")
    m = re.search(r"(\d+)$", str(raw))
    return int(m.group(1)) if m else None

available = sorted(n for n in (version_number(v) for v in project.versions()) if n is not None)
print(f"Published versions available: {available}")
if not available:
    raise RuntimeError("No published dataset versions were returned for this project.")
if DATASET_VERSION is None:
    DATASET_VERSION = available[-1]
elif DATASET_VERSION not in available:
    raise ValueError(f"Version {DATASET_VERSION} not found. Choose one of {available}.")
print(f"Using dataset version: {DATASET_VERSION}")

dataset = project.version(DATASET_VERSION).download(EXPORT_FORMAT, location=f"/content/datasets/{PROJECT}-v{DATASET_VERSION}")
DATA_DIR = Path(dataset.location)
DATA_YAML = DATA_DIR / "data.yaml"
print(f"Dataset location: {DATA_DIR}")
""")
code(r"""
# Normalise data.yaml paths to absolute paths and validate them.
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

with open(DATA_YAML) as f:
    data_cfg = yaml.safe_load(f)

names = data_cfg["names"]
CLASS_NAMES = {i: n for i, n in enumerate(names)} if isinstance(names, list) else {int(k): v for k, v in names.items()}
print("Classes (index → name) read from data.yaml:")
for i, n in CLASS_NAMES.items():
    print(f"  {i}: {n}")

def resolve_split(key):
    aliases = ("val", "valid", "validation") if key == "val" else (key,)
    root = Path(data_cfg.get("path") or DATA_DIR)
    if not root.is_absolute():
        root = DATA_DIR / root
    candidates = []
    for alias in aliases:
        rel = data_cfg.get(alias)
        if rel:
            if not isinstance(rel, str):
                raise ValueError(f"Expected one image-directory path for '{alias}', got {rel!r}.")
            candidates.extend((root / rel, DATA_DIR / rel))
    for base in (DATA_DIR, root):
        for alias in aliases:
            candidates.extend((base / alias / "images", base / "images" / alias))
    for c in candidates:
        c = c.resolve()
        if c.is_dir() and any(p.is_file() and p.suffix.lower() in IMG_EXT for p in c.iterdir()):
            return c
    return None

SPLITS = {}
for key in ("train", "val", "test"):
    p = resolve_split(key)
    if p is not None:
        SPLITS[key] = p
        data_cfg[key] = str(p)
    elif key in data_cfg:
        print(f"⚠️  '{key}' listed in data.yaml but its folder was not found — removing it.")
        data_cfg.pop(key)

if "train" in SPLITS and "val" not in SPLITS:
    # Some Roboflow versions export training data only. Keep the download intact.
    import random, shutil, tempfile
    from collections import defaultdict

    source = SPLITS["train"]
    images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
    groups = defaultdict(list)
    for image in images:
        # Roboflow variants commonly share the name before '.rf.'.
        groups[image.stem.split(".rf.")[0]].append(image)
    group_names = sorted(groups)
    if len(group_names) < 2:
        raise RuntimeError("Need at least two source-image groups to create a validation holdout.")
    missing_labels = [p.name for p in images if not (source.parent / "labels" / f"{p.stem}.txt").is_file()]
    if missing_labels:
        raise RuntimeError(f"Cannot split: {len(missing_labels)} images lack labels; examples: {missing_labels[:5]}")
    random.Random(42).shuffle(group_names)
    val_count = min(len(group_names) - 1, max(1, round(len(group_names) * 0.2)))
    val_groups = set(group_names[:val_count])
    holdout = Path(tempfile.mkdtemp(prefix="holdout-seed42-", dir=DATA_DIR))
    for split in ("train", "val"):
        (holdout / split / "images").mkdir(parents=True)
        (holdout / split / "labels").mkdir()
    for group, members in groups.items():
        split = "val" if group in val_groups else "train"
        for image in members:
            shutil.copy2(image, holdout / split / "images" / image.name)
            label = source.parent / "labels" / f"{image.stem}.txt"
            shutil.copy2(label, holdout / split / "labels" / label.name)
    for split in ("train", "val"):
        SPLITS[split] = (holdout / split / "images").resolve()
        data_cfg[split] = str(SPLITS[split])
    data_cfg["split_provenance"] = {
        "method": "local source-filename grouped holdout", "seed": 42,
        "validation_group_fraction": 0.2, "source_train": str(source),
    }
    print(f"Created a local 80/20 source-group holdout in {holdout}; original download unchanged.")
    print("This is not an official Roboflow validation split. Related frames or renamed duplicates may still leak.")

if "train" not in SPLITS or "val" not in SPLITS:
    folders = sorted(str(p.relative_to(DATA_DIR)) for p in DATA_DIR.rglob("*") if p.is_dir())
    raise RuntimeError(
        f"Non-empty train/val image directories are required. Found: {SPLITS}. "
        f"Directories under {DATA_DIR}: {folders}. "
        "If validation images are absent, regenerate the Roboflow version with separate "
        "training and validation splits, then download it again. Do not reuse training "
        "images as validation. data.yaml has not been rewritten."
    )

if len(set(SPLITS.values())) != len(SPLITS):
    raise RuntimeError(f"Dataset splits resolve to the same directory: {SPLITS}. Use separate splits.")

data_cfg.pop("valid", None)
data_cfg.pop("validation", None)
data_cfg["path"] = str(DATA_DIR)
with open(DATA_YAML, "w") as f:
    yaml.safe_dump(data_cfg, f, sort_keys=False)

def label_path(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")

SPLIT_IMAGES = {k: sorted(p for p in v.iterdir() if p.suffix.lower() in IMG_EXT) for k, v in SPLITS.items()}
for k, imgs in SPLIT_IMAGES.items():
    missing = sum(1 for p in imgs if not label_path(p).exists())
    print(f"{k:5s}: {len(imgs):5d} images | {missing} without a label file")
if "test" not in SPLITS:
    print("test : no test split in this dataset version")
""")
code(r"""
# Visual check: random annotated training images.
import cv2, matplotlib.pyplot as plt

def class_color(name):
    n = name.lower()
    if "unripe" in n: return (255, 140, 0)    # orange (RGB)
    if "ripe" in n:   return (40, 200, 70)    # green
    return (160, 160, 160)

def read_labels(img_path):
    lp = label_path(img_path)
    rows = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append((int(float(parts[0])), *map(float, parts[1:5])))
    return rows

def draw_gt(img_path):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    for c, cx, cy, bw, bh in read_labels(img_path):
        x1, y1, x2, y2 = int((cx-bw/2)*w), int((cy-bh/2)*h), int((cx+bw/2)*w), int((cy+bh/2)*h)
        name = CLASS_NAMES.get(c, f"id{c}")
        col = class_color(name)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, name, (x1, max(12, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img

random.seed(42)
sample = random.sample(SPLIT_IMAGES["train"], min(8, len(SPLIT_IMAGES["train"])))
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, p in zip(axes.flat, sample):
    ax.imshow(draw_gt(p)); ax.set_title(p.name[:30], fontsize=8); ax.axis("off")
for ax in axes.flat[len(sample):]:
    ax.axis("off")
plt.suptitle("Ground-truth annotations (green = ripe, orange = unripe)"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- Cell 4
md("## 3 · Dataset analysis")
code(r"""
import pandas as pd
import numpy as np

rows, warnings = [], []
for split, imgs in SPLIT_IMAGES.items():
    for p in imgs:
        labels = read_labels(p)
        if not label_path(p).exists():
            warnings.append(f"{split}/{p.name}: missing label file")
        elif not labels:
            rows.append({"split": split, "image": p.name, "class_id": None, "w": None, "h": None})
        for c, cx, cy, bw, bh in labels:
            if c not in CLASS_NAMES:
                warnings.append(f"{split}/{p.name}: class id {c} not in data.yaml")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                warnings.append(f"{split}/{p.name}: box outside normalised range")
            rows.append({"split": split, "image": p.name, "class_id": c, "w": bw, "h": bh})

df = pd.DataFrame(rows)
boxes = df.dropna(subset=["class_id"]).copy()
boxes["class_id"] = boxes["class_id"].astype(int)
boxes["class"] = boxes["class_id"].map(CLASS_NAMES)

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
split_counts = pd.Series({k: len(v) for k, v in SPLIT_IMAGES.items()})
split_counts.plot.bar(ax=axes[0], color="#5b8def"); axes[0].set_title("Images per split")
for i, v in enumerate(split_counts): axes[0].text(i, v, str(v), ha="center", va="bottom")

dist = boxes.groupby(["split", "class"]).size().unstack(fill_value=0)
dist.plot.bar(ax=axes[1], color=[np.array(class_color(c))/255 for c in dist.columns])
axes[1].set_title("Boxes per class and split")

axes[2].scatter(boxes["w"], boxes["h"], s=2, alpha=0.3,
                c=[np.array(class_color(c))/255 for c in boxes["class"]])
axes[2].set_xlabel("box width (normalised)"); axes[2].set_ylabel("box height (normalised)")
axes[2].set_title("Bounding-box size distribution")
plt.tight_layout(); plt.show()

print("Boxes per class (all splits):")
total_per_class = boxes["class"].value_counts()
print(total_per_class.to_string())
print(f"\nMean boxes per image: {len(boxes) / max(1, split_counts.sum()):.2f}")

# Health warnings
empty = df[df["class_id"].isna()]
if len(empty):
    warnings.append(f"{len(empty)} images have an empty label file (background images)")
tiny = boxes[(boxes["w"] * boxes["h"]) < 0.0005]
if len(tiny):
    warnings.append(f"{len(tiny)} very small boxes (<0.05% of image area) — may be hard to learn")
if len(total_per_class) >= 2:
    ratio = total_per_class.max() / max(1, total_per_class.min())
    if ratio > 3:
        warnings.append(f"Class imbalance: '{total_per_class.idxmax()}' has {ratio:.1f}× more boxes than '{total_per_class.idxmin()}'")
    else:
        print(f"Class balance ratio (max/min): {ratio:.2f} — acceptable")
missing_classes = set(CLASS_NAMES.values()) - set(total_per_class.index)
if missing_classes:
    warnings.append(f"Classes with zero annotations: {sorted(missing_classes)}")

print("\nDataset health:")
if warnings:
    for w in warnings[:30]: print("  ⚠️ ", w)
    if len(warnings) > 30: print(f"  ... and {len(warnings) - 30} more")
else:
    print("  ✅ no problems found")
""")

# ---------------------------------------------------------------- Cell 5
md(r"""
## 4 · Model configuration

Colour is the main ripeness cue, so hue augmentation is **off** (`hsv_h=0`) and saturation/value jitter is mild.
Geometric augmentation (flips, small rotations, scale, mosaic) is kept because conveyor footage varies in
position and scale but not in colour.
""")
code(r"""
from datetime import datetime, timezone

MODEL_ARCH = "yolo11n.pt"   # switch to "yolo11s.pt" for higher accuracy if VRAM allows
RUN_DIR = Path("/content/runs/tomatosort")
RUN_NAME = f"{Path(MODEL_ARCH).stem}_v{DATASET_VERSION}"

TRAIN_CFG = dict(
    data=str(DATA_YAML),
    epochs=80,
    imgsz=640,
    batch=-1 if CUDA else 8,       # -1 = AutoBatch (~60% of VRAM)
    patience=15,
    device=DEVICE,
    seed=42,
    deterministic=True,
    plots=True,
    save=True,
    project=str(RUN_DIR),
    name=RUN_NAME,
    exist_ok=True,
    workers=2 if IN_COLAB else 4,
    cos_lr=True,
    close_mosaic=10,
    # colour-preserving augmentation
    hsv_h=0.0, hsv_s=0.35, hsv_v=0.30,
    degrees=5.0, translate=0.1, scale=0.4, shear=0.0,
    fliplr=0.5, flipud=0.1, mosaic=1.0, mixup=0.0,
)
for k, v in TRAIN_CFG.items():
    print(f"{k:14s} {v}")
""")

# ---------------------------------------------------------------- Cell 6
md(r"""
## 5 · Train

Set `RESUME = True` to continue an interrupted run from its `last.pt`.
If CUDA runs out of memory the batch size is halved and training restarts automatically.
""")
code(r"""
import gc
from ultralytics import YOLO

RESUME = False
TRAIN_START = datetime.now(timezone.utc)
SAVE_DIR = RUN_DIR / RUN_NAME
last_ckpt = SAVE_DIR / "weights" / "last.pt"

def is_oom(err):
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()

if RESUME and last_ckpt.exists():
    print(f"Resuming from {last_ckpt}")
    YOLO(str(last_ckpt)).train(resume=True)
else:
    if RESUME:
        print(f"⚠️  RESUME=True but {last_ckpt} does not exist — starting a fresh run.")
    batch = TRAIN_CFG["batch"]
    while True:
        model = None
        try:
            model = YOLO(MODEL_ARCH)
            model.train(**{**TRAIN_CFG, "batch": batch})
            break
        except Exception as e:
            if not is_oom(e):
                raise
            model = None; gc.collect(); torch.cuda.empty_cache()
            batch = 16 if batch == -1 else batch // 2
            if batch < 2:
                raise RuntimeError("Out of memory even at batch 2. Lower imgsz or use yolo11n.") from e
            print(f"⚠️  CUDA out of memory — retrying with batch={batch}")

BEST_PT = SAVE_DIR / "weights" / "best.pt"
LAST_PT = SAVE_DIR / "weights" / "last.pt"
print(f"\nArtifacts in {SAVE_DIR}")
print(f"best.pt exists: {BEST_PT.exists()} | last.pt exists: {LAST_PT.exists()}")
""")

# ---------------------------------------------------------------- Cell 7
md("## 6 · Training performance\nAll numbers below are read from this run's `results.csv` and plots — nothing is hard-coded.")
code(r"""
from IPython.display import Image as IPImage, display

res = pd.read_csv(SAVE_DIR / "results.csv")
res.columns = [c.strip() for c in res.columns]

def col(name):
    return next((c for c in res.columns if c.endswith(name)), None)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for ax, key in zip(axes[0], ["box_loss", "cls_loss", "dfl_loss"]):
    for split in ("train", "val"):
        c = f"{split}/{key}"
        if c in res: ax.plot(res["epoch"], res[c], label=split)
    ax.set_title(key); ax.legend(); ax.grid(alpha=0.3)
for ax, key in zip(axes[1], ["precision(B)", "recall(B)", "mAP50(B)"]):
    c = col(key)
    if c: ax.plot(res["epoch"], res[c], color="#e4572e")
    ax.set_title(key); ax.grid(alpha=0.3)
c95 = col("mAP50-95(B)")
if c95: axes[1][2].plot(res["epoch"], res[c95], color="#29335c", label="mAP50-95"); axes[1][2].legend()
plt.tight_layout(); plt.show()

best_epoch = res.loc[res[c95].idxmax()] if c95 else res.iloc[-1]
print(f"Epochs run: {len(res)} | best epoch (by mAP50-95): {int(best_epoch['epoch'])}")
for key in ["precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"]:
    c = col(key)
    if c: print(f"  {key:13s} {best_epoch[c]:.4f}")

for pattern in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "*PR_curve.png", "*F1_curve.png"]:
    for p in sorted(SAVE_DIR.glob(pattern)):
        print(p.name); display(IPImage(filename=str(p), width=900))
""")

# ---------------------------------------------------------------- Cell 8
md(r"""
## 7 · Validation (and test, if available)

`best.pt` is evaluated on the validation split. If the dataset has a separate test split, a final evaluation runs on it —
the training configuration is **not** changed in response to test results.
""")
code(r"""
best_model = YOLO(str(BEST_PT))

def per_class_table(metrics):
    box = metrics.box
    out = []
    for i, cls_idx in enumerate(box.ap_class_index):
        out.append({"class": CLASS_NAMES[int(cls_idx)], "precision": box.p[i], "recall": box.r[i],
                    "mAP50": box.ap50[i], "mAP50-95": box.ap[i]})
    return pd.DataFrame(out).set_index("class").round(4)

def summary(metrics):
    return {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)}

val_metrics = best_model.val(data=str(DATA_YAML), split="val", imgsz=TRAIN_CFG["imgsz"], device=DEVICE, plots=False)
VAL_SUMMARY = summary(val_metrics)
print("Validation:", {k: round(v, 4) for k, v in VAL_SUMMARY.items()})
display(per_class_table(val_metrics))

TEST_SUMMARY = None
if "test" in SPLITS:
    test_metrics = best_model.val(data=str(DATA_YAML), split="test", imgsz=TRAIN_CFG["imgsz"], device=DEVICE, plots=False)
    TEST_SUMMARY = summary(test_metrics)
    print("Test (final, held-out):", {k: round(v, 4) for k, v in TEST_SUMMARY.items()})
    display(per_class_table(test_metrics))
else:
    print("ℹ️  This dataset version has no independent test split. The validation split was also used for "
          "early stopping / best-checkpoint selection, so the validation numbers are slightly optimistic.")
""")
code(r"""
# Correct vs. incorrect sample predictions on the validation split.
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)

def evaluate_image(p, conf=0.4):
    img = cv2.imread(str(p)); h, w = img.shape[:2]
    gts = [(c, ((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h)) for c, cx, cy, bw, bh in read_labels(p)]
    r = best_model.predict(img, conf=conf, verbose=False, device=DEVICE)[0]
    preds = [(int(c), float(s), b.tolist()) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    used, errors = set(), 0
    for pc, ps, pb in preds:
        match = max(((i, iou(pb, gb)) for i, (gc, gb) in enumerate(gts) if i not in used), key=lambda t: t[1], default=(None, 0))
        if match[1] >= 0.5:
            used.add(match[0]); errors += gts[match[0]][0] != pc
        else:
            errors += 1          # false positive
    errors += len(gts) - len(used)  # missed tomatoes
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for pc, ps, pb in preds:
        name = CLASS_NAMES[pc]; colr = class_color(name)
        x1, y1, x2, y2 = map(int, pb)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colr, 2)
        cv2.putText(vis, f"{name} {ps:.2f}", (x1, max(12, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colr, 2)
    return errors, len(gts), vis

random.seed(0)
pool = random.sample(SPLIT_IMAGES["val"], min(60, len(SPLIT_IMAGES["val"])))
good, bad = [], []
for p in pool:
    errs, n, vis = evaluate_image(p)
    (bad if errs else good).append((p.name, errs, n, vis))

for title, items in (("Fully correct predictions", good[:4]), ("Predictions with errors (miss / false positive / wrong class)", bad[:4])):
    if not items:
        print(f"{title}: none in the sampled images"); continue
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 5))
    for ax, (name, errs, n, vis) in zip(np.atleast_1d(axes), items):
        ax.imshow(vis); ax.set_title(f"{name[:22]}\nGT={n} errors={errs}", fontsize=8); ax.axis("off")
    plt.suptitle(title); plt.tight_layout(); plt.show()
print(f"Sampled {len(pool)} val images: {len(good)} fully correct, {len(bad)} with at least one error (IoU≥0.5, conf≥0.4)")
""")

# ---------------------------------------------------------------- Cell 9
md("## 8 · Test on your own images\nUpload one or more images. Green = ripe, orange = unripe.")
code(r"""
def annotate(frame_bgr, result, min_conf=0.0):
    out = frame_bgr.copy()
    counts = {n: 0 for n in CLASS_NAMES.values()}
    for c, s, b in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
        if float(s) < min_conf:
            continue
        name = CLASS_NAMES[int(c)]
        counts[name] += 1
        r, g, bl = class_color(name)
        x1, y1, x2, y2 = map(int, b.tolist())
        cv2.rectangle(out, (x1, y1), (x2, y2), (bl, g, r), 2)
        label = f"{name} {float(s):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (bl, g, r), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    summary_txt = "  ".join(f"{k}: {v}" for k, v in counts.items())
    cv2.putText(out, summary_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, summary_txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out, counts

CONF = 0.4
if IN_COLAB:
    from google.colab import files
    uploaded = files.upload()
    for fname in uploaded:
        img = cv2.imread(fname)
        if img is None:
            print(f"Skipping {fname}: not a readable image"); continue
        res_ = best_model.predict(img, conf=CONF, verbose=False, device=DEVICE)[0]
        vis, counts = annotate(img, res_)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original")
        axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); axes[1].set_title(f"Detections {counts}")
        for a in axes: a.axis("off")
        plt.tight_layout(); plt.show()
else:
    print("Upload widget requires Colab. Set a path manually and call best_model.predict(...).")
""")

# ---------------------------------------------------------------- Cell 10
md(r"""
## 9 · Test on a video

Upload an MP4 (for example a conveyor-belt clip). Every frame is processed and an annotated, browser-playable H.264 MP4 is saved
and offered for download. This is offline notebook testing — the live dashboard handles webcams and streams.
""")
code(r"""
import subprocess
from IPython.display import HTML
from base64 import b64encode

def process_video(src, dst, conf=CONF):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tmp = dst + ".raw.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        r = best_model.predict(frame, conf=conf, verbose=False, device=DEVICE)[0]
        writer.write(annotate(frame, r)[0]); i += 1
        if i % 50 == 0: print(f"  {i}/{n} frames")
    cap.release(); writer.release()
    # re-encode to H.264 so browsers can play it
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", dst], check=True)
    os.remove(tmp)
    print(f"Saved {dst} ({i} frames)")

if IN_COLAB:
    up = files.upload()
    for fname in up:
        if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            print(f"Skipping {fname}: not a video"); continue
        out_path = f"/content/annotated_{Path(fname).stem}.mp4"
        process_video(fname, out_path)
        if os.path.getsize(out_path) < 40e6:
            display(HTML(f'<video width=800 controls src="data:video/mp4;base64,{b64encode(open(out_path,"rb").read()).decode()}"></video>'))
        files.download(out_path)
""")

# ---------------------------------------------------------------- Cell 11
md("## 10 · Export model and metadata")
code(r"""
import json, shutil

EXPORT_ONNX = True
SAVE_TO_DRIVE = False

EXPORT_DIR = Path("/content/tomatosort_export")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy(BEST_PT, EXPORT_DIR / "best.pt")
if LAST_PT.exists():
    shutil.copy(LAST_PT, EXPORT_DIR / "last.pt")

if EXPORT_ONNX:
    try:
        onnx_path = YOLO(str(BEST_PT)).export(format="onnx", imgsz=TRAIN_CFG["imgsz"], dynamic=False, simplify=True)
        shutil.copy(onnx_path, EXPORT_DIR / "model.onnx")
    except Exception as e:
        print(f"⚠️  ONNX export failed ({e}); continuing without it.")

metadata = {
    "app": "TOMATO//SORT AI",
    "model_architecture": Path(MODEL_ARCH).stem,
    "pretrained_weights": MODEL_ARCH,
    "class_map": {str(k): v for k, v in CLASS_NAMES.items()},
    "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": DATASET_VERSION,
                "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{DATASET_VERSION}",
                "images": {k: len(v) for k, v in SPLIT_IMAGES.items()}},
    "imgsz": TRAIN_CFG["imgsz"],
    "epochs_configured": TRAIN_CFG["epochs"],
    "epochs_run": int(len(res)),
    "training_date": TRAIN_START.isoformat(),
    "ultralytics_version": ultralytics.__version__,
    "metrics": {"val": VAL_SUMMARY, "test": TEST_SUMMARY},
}
(EXPORT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

training_metrics = {
    "per_epoch": res.to_dict(orient="records"),
    "final_val": VAL_SUMMARY,
    "final_test": TEST_SUMMARY,
    "per_class_val": per_class_table(val_metrics).reset_index().to_dict(orient="records"),
}
(EXPORT_DIR / "training_metrics.json").write_text(json.dumps(training_metrics, indent=2, default=float))

print("Exported files:")
for p in sorted(EXPORT_DIR.iterdir()):
    print(f"  {p.name:24s} {p.stat().st_size/1e6:7.2f} MB")

zip_path = shutil.make_archive("/content/tomatosort_model", "zip", EXPORT_DIR)
if IN_COLAB:
    files.download(str(EXPORT_DIR / "best.pt"))
    files.download(zip_path)

if SAVE_TO_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    dest = Path("/content/drive/MyDrive/TomatoSort_AI") / RUN_NAME
    shutil.copytree(EXPORT_DIR, dest, dirs_exist_ok=True)
    print(f"Copied to {dest}")
""")

# ---------------------------------------------------------------- Cell 12
md(r"""
## 11 · Deploy to the local TOMATO//SORT AI app

1. Copy the downloaded files into the project:

```
tomato-sort-ai/
└── backend/models/
    ├── best.pt                 ← required
    └── model_metadata.json     ← recommended (shown in the dashboard)
```

2. Start the backend (from `tomato-sort-ai/backend`):

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

3. Start the dashboard (from `tomato-sort-ai/frontend`): `npm install && npm run dev`, then open http://localhost:5173.

**Input / output.** The model takes a BGR or RGB frame of any size (letterboxed to `imgsz` internally) and returns, per detection,
`xyxy` pixel coordinates, a confidence in `[0, 1]` and a class id that maps to the names in `model_metadata.json`.
""")
code(r"""
# Minimal usage example (same calls the backend makes)
m = YOLO(str(EXPORT_DIR / "best.pt"))
print("Classes:", m.names)
example = SPLIT_IMAGES["val"][0]
r = m.predict(str(example), conf=0.4, verbose=False, device=DEVICE)[0]
for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
    print({"class": m.names[int(c)], "confidence": round(float(s), 3), "xyxy": [round(v, 1) for v in b.tolist()]})
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "name": "TomatoSort_AI_Training.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
