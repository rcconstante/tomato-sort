import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { EventLog } from "./components/EventLog";
import { SettingsPanel } from "./components/SettingsPanel";
import { SourceControls } from "./components/SourceControls";
import { LatestInspection, StatsPanel } from "./components/StatsPanel";
import { VideoPanel } from "./components/VideoPanel";
import type { InspectionLine, ModelStatus, Settings } from "./types";
import { useTelemetry } from "./useTelemetry";

function mergePatch(s: Settings, patch: Record<string, unknown>): Settings {
  const next: Record<string, unknown> = { ...s };
  for (const [k, v] of Object.entries(patch)) {
    const cur = (s as unknown as Record<string, unknown>)[k];
    next[k] = v && typeof v === "object" && cur && typeof cur === "object" ? { ...cur, ...v } : v;
  }
  return next as unknown as Settings;
}

export default function App() {
  const { connected, status, model: liveModel, events } = useTelemetry();
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [editingLine, setEditingLine] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const pending = useRef<Record<string, unknown>>({});
  const timer = useRef<number>();

  const loadModel = useCallback(() => api.model().then(setModel).catch(() => setModel(null)), []);

  useEffect(() => {
    if (!connected) return;
    loadModel();
    api.settings().then(setSettings).catch(() => undefined);
  }, [connected, loadModel]);

  // keep model info in sync if the backend reports a change (e.g. reload from another tab)
  useEffect(() => {
    if (liveModel && model && liveModel.installed !== model.installed) loadModel();
  }, [liveModel, model, loadModel]);

  const changeSettings = useCallback((patch: Record<string, unknown>) => {
    setSettings((s) => (s ? mergePatch(s, patch) : s));
    for (const [k, v] of Object.entries(patch)) {
      const prev = pending.current[k];
      pending.current[k] = v && typeof v === "object" && prev && typeof prev === "object" ? { ...prev, ...v } : v;
    }
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      const body = pending.current;
      pending.current = {};
      api
        .patchSettings(body)
        .then((s) => {
          setSettings(s);
          setSettingsError(null);
        })
        .catch((e: Error) => setSettingsError(e.message));
    }, 200);
  }, []);

  const classes = model ? Object.values(model.classes) : [];
  const modelReady = !!model?.installed;
  const meta = model?.metadata;
  const valMap = meta?.metrics?.test?.mAP50 ?? meta?.metrics?.val?.mAP50;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">TOMATO</span>
          <span className="brand-slash">//</span>
          <span className="brand-mark">SORT</span>
          <span className="brand-ai">AI</span>
          <span className="brand-sub">Conveyor ripeness inspection</span>
        </div>
        <div className="topbar-right">
          {modelReady && (
            <span className="pill">
              {meta?.model_architecture ?? "best.pt"}
              {meta?.dataset?.version != null && ` · dataset v${meta.dataset.version}`}
              {valMap != null && ` · mAP50 ${(valMap * 100).toFixed(1)}%`}
              {` · ${model!.device}`}
            </span>
          )}
          <span className={connected ? "pill ok" : "pill bad"}>
            <span className="dot" />
            {connected ? "Backend connected" : "Backend offline"}
          </span>
        </div>
      </header>

      {!connected && (
        <div className="banner">
          Can't reach the inference server on port 8000. Start it from <code>backend/</code> with{" "}
          <code>uvicorn app.main:app --port 8000</code>.
        </div>
      )}
      {connected && model && !model.installed && (
        <div className="banner">
          <strong>Model not installed.</strong> {model.error}
          <button className="btn small" onClick={() => api.reloadModel().then(setModel)}>
            Reload model
          </button>
        </div>
      )}

      <main className="layout">
        <section className="col-main">
          <div className="panel video-panel">
            <VideoPanel
              status={status}
              line={settings?.line ?? null}
              editing={editingLine}
              onLineCommit={(line: InspectionLine) => changeSettings({ line })}
            />
            {status?.source.name && (
              <div className="source-meta muted small">
                {status.source.name}
                {status.source.width ? ` · ${status.source.width}×${status.source.height}` : ""}
                {status.source.fps ? ` · ${status.source.fps} fps` : ""}
                {` · frame ${status.frame_index}`}
                {status.loops > 0 && ` · loop ${status.loops + 1}`}
              </div>
            )}
          </div>
          <SourceControls status={status} modelReady={modelReady} />
          <EventLog events={events} sessionId={status?.session_id ?? null} />
        </section>

        <aside className="col-side">
          <LatestInspection latest={events[0] ?? null} />
          <StatsPanel status={status} classes={classes} />
          {settings && (
            <SettingsPanel
              settings={settings}
              classes={classes}
              editingLine={editingLine}
              onToggleEditLine={() => setEditingLine((v) => !v)}
              onChange={changeSettings}
            />
          )}
          {settingsError && <div className="error">{settingsError}</div>}
        </aside>
      </main>
    </div>
  );
}
