import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { EventLog } from "./components/EventLog";
import { SettingsPanel } from "./components/SettingsPanel";
import { SourceControls } from "./components/SourceControls";
import { InspectionSummary } from "./components/InspectionSummary";
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
  const presentation = new URLSearchParams(window.location.search).has("presentation");

  return (
    <div className={presentation ? "app presentation" : "app"}>
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

      <main className="inspection-shell">
        <section className="inspection-card" aria-label="Tomato ripeness inspection">
          <header className="inspection-heading"><h1>Tomato Ripeness <span>Inspection</span></h1></header>
            <VideoPanel
              status={status}
              line={settings?.line ?? null}
              editing={editingLine && settings?.count_trigger === "line"}
              onLineCommit={(line: InspectionLine) => changeSettings({ line })}
            />
          <InspectionSummary latest={events[0] ?? null} status={status} />
        </section>
        {status?.error && <div className="error" role="alert">{status.error}</div>}
        {!presentation && <details className="setup-panel">
          <summary>Setup & controls</summary>
          <div className="setup-grid">
          <SourceControls status={status} modelReady={modelReady} />
          {settings && (
            <SettingsPanel
              settings={settings}
              classes={classes}
              editingLine={editingLine}
              onToggleEditLine={() => setEditingLine((v) => !v)}
              onChange={changeSettings}
            />
          )}
          </div>
          {settingsError && <div className="error">{settingsError}</div>}
          <EventLog events={events} sessionId={status?.session_id ?? null} />
        </details>}
      </main>
    </div>
  );
}
