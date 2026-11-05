# Model folder

Put the files exported by `notebooks/TomatoSort_AI_Training.ipynb` (cell 10, "Export model and metadata") here:

```
backend/models/
├── best.pt                 required: the trained YOLO detector
└── model_metadata.json     recommended: shown in the dashboard (dataset version, metrics)
```

The server never falls back to a different model or simulated detections. Without `best.pt` the dashboard
shows a "model not installed" banner and inspection cannot start.

To use a model stored elsewhere, set `TOMATOSORT_MODEL=C:\path\to\best.pt` before starting the server.
