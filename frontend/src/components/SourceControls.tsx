import { useRef, useState } from "react";
import { api } from "../api";
import type { PipelineStatus, SourceType } from "../types";

interface Props {
  status: PipelineStatus | null;
  modelReady: boolean;
}

const TABS: { id: SourceType; label: string }[] = [
  { id: "upload", label: "Upload MP4" },
  { id: "webcam", label: "Webcam" },
  { id: "file", label: "Local file" },
  { id: "rtsp", label: "RTSP / URL" },
];

export function SourceControls({ status, modelReady }: Props) {
  const [tab, setTab] = useState<SourceType>("upload");
  const [upload, setUpload] = useState<{ id: string; name: string } | null>(null);
  const [webcam, setWebcam] = useState("0");
  const [filePath, setFilePath] = useState("");
  const [rtsp, setRtsp] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const state = status?.state ?? "idle";
  const active = state === "running" || state === "paused";
  const value = { upload: upload?.id ?? "", webcam, file: filePath, rtsp }[tab];

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const onFile = (f: File | undefined) => {
    if (!f) return;
    run("Uploading…", async () => {
      const res = await api.upload(f);
      setUpload({ id: res.upload_id, name: res.name });
    });
  };

  return (
    <div className="panel source-panel">
      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      <div className="source-input">
        {tab === "upload" && (
          <>
            <input ref={fileInput} type="file" accept="video/*,.mp4,.mov,.avi,.mkv" hidden onChange={(e) => onFile(e.target.files?.[0])} />
            <button className="btn ghost" onClick={() => fileInput.current?.click()}>
              Choose video…
            </button>
            <span className="muted truncate">{upload ? upload.name : "No video uploaded"}</span>
          </>
        )}
        {tab === "webcam" && (
          <label className="field-inline">
            Camera index
            <input type="number" min={0} max={16} value={webcam} onChange={(e) => setWebcam(e.target.value)} />
          </label>
        )}
        {tab === "file" && (
          <input className="grow" placeholder="C:\videos\conveyor.mp4" value={filePath} onChange={(e) => setFilePath(e.target.value)} />
        )}
        {tab === "rtsp" && (
          <input className="grow" placeholder="rtsp://user:pass@192.168.1.20:554/stream1" value={rtsp} onChange={(e) => setRtsp(e.target.value)} />
        )}
      </div>

      <div className="controls">
        <button className="btn primary" disabled={!modelReady || !value || !!busy} onClick={() => run("Starting…", () => api.start(tab, value))}>
          {active ? "Restart" : "Start"}
        </button>
        {state === "paused" ? (
          <button className="btn" onClick={() => run("Resuming…", api.resume)}>
            Resume
          </button>
        ) : (
          <button className="btn" disabled={state !== "running"} onClick={() => run("Pausing…", api.pause)}>
            Pause
          </button>
        )}
        <button className="btn" disabled={!active} onClick={() => run("Stopping…", api.stop)}>
          Stop
        </button>
        <button className="btn danger" disabled={state === "idle"} onClick={() => run("Resetting…", api.reset)}>
          Reset
        </button>
        {busy && <span className="muted">{busy}</span>}
      </div>

      {!modelReady && <div className="note">Inspection is disabled until a trained model is installed.</div>}
      {(error || status?.error) && <div className="error">{error ?? status?.error}</div>}
    </div>
  );
}
