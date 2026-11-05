import { capitalize, colorOf } from "../format";
import type { InspectionEvent, PipelineStatus } from "../types";

export function InspectionSummary({ latest, status }: { latest: InspectionEvent | null; status: PipelineStatus | null }) {
  const labels = status?.counts.labels ?? {};
  return (
    <footer className="inspection-summary">
      <div className="inspection-crop" style={{ borderColor: latest ? colorOf(latest.label) : undefined }}>
        {latest?.crop ? <img src={`data:image/jpeg;base64,${latest.crop}`} alt={`Latest inspected tomato: ${latest.label}`} />
          : <span className="muted small">Awaiting inspection</span>}
      </div>
      <div className="inspection-latest">
        <p className="eyebrow">Latest inspection</p>
        <h2 style={{ color: latest ? colorOf(latest.label) : undefined }}>{latest ? capitalize(latest.label) : "Ready to inspect"}</h2>
        {latest && <p className="muted">Tomato #{latest.number} · track #{latest.track_id}</p>}
        {latest?.label === "uncertain" && latest.reason && <p className="muted small">{latest.reason}</p>}
        <p className="inspection-wordmark">TOMATO<span>//</span>SORT <small>AI</small></p>
      </div>
      <div className="inspection-totals" aria-live="off">
        <strong className="inspection-total">{status?.counts.total ?? 0}</strong>
        <p className="muted">tomatoes counted</p>
        <p className="count-row"><i className="dot" style={{ background: "var(--ripe)" }} /><strong>{labels.ripe ?? 0}</strong> Ripe</p>
        <p className="count-row"><i className="dot" style={{ background: "var(--unripe)" }} /><strong>{labels.unripe ?? 0}</strong> Unripe</p>
        {(labels.uncertain ?? 0) > 0 && <p className="count-row"><i className="dot" /><strong>{labels.uncertain}</strong> Uncertain</p>}
      </div>
    </footer>
  );
}
