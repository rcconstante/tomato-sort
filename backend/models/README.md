# Model folder

This folder includes a trained detector and its metadata. To replace them, use the export section of [the training notebook](../../notebooks/TomatoSort_AI_Training.ipynb):

```
backend/models/
├── best.pt                 required: the trained YOLO detector
└── model_metadata.json     recommended: recorded dataset version and validation metrics
```

The server never falls back to a different model or simulated detections. Without `best.pt` the dashboard
shows a "model not installed" banner and inspection cannot start.

To use a model stored elsewhere, set `TOMATOSORT_MODEL=C:\path\to\best.pt` before starting the server.
