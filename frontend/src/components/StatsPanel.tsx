import { CATEGORY_COLOR, DECISION_COLOR, capitalize, categoryOf, colorOf, pct } from "../format";
import type { Decision, InspectionEvent, PipelineStatus } from "../types";

interface Props {
  status: PipelineStatus | null;
  classes: string[];
}

export function LatestInspection({ latest }: { latest: InspectionEvent | null }) {
  if (!latest) {
    return (
      <div className="panel latest empty">
        <div className="eyebrow">Latest inspection</div>
        <div className="muted">Tomatoes are graded when they cross the inspection line.</div>
      </div>
    );
  }
  const color = colorOf(latest.label);
  return (
    <div className="panel latest">
      <div className="latest-crop" style={{ borderColor: color }}>
        {latest.crop ? <img src={`data:image/jpeg;base64,${latest.crop}`} alt={`Tomato ${latest.number}`} /> : <div className="muted">no crop</div>}
      </div>
      <div className="latest-body">
        <div className="eyebrow">Latest inspection</div>
        <div className="latest-title" style={{ color }}>
          {capitalize(latest.label)}
          <span className="decision-chip" style={{ background: DECISION_COLOR[latest.decision] }}>
            {latest.decision}
          </span>
        </div>
        <div className="muted small">
          tomato #{latest.number} · track #{latest.track_id} · {latest.observations} frames
        </div>
        <div className="meter-label">
          <span>confidence</span>
          <span>{pct(latest.confidence)}</span>
        </div>
        <div className="meter">
          <div style={{ width: pct(latest.confidence), background: color }} />
        </div>
        <div className="meter-label">
          <span>vote agreement</span>
          <span>{pct(latest.vote_share)}</span>
        </div>
        <div className="meter">
          <div style={{ width: pct(latest.vote_share), background: "var(--accent)" }} />
        </div>
        {latest.reason && <div className="note small">{latest.reason}</div>}
      </div>
    </div>
  );
}

export function StatsPanel({ status, classes }: Props) {
  const counts = status?.counts;
  const total = counts?.total ?? 0;
  const labels = Array.from(new Set([...classes, ...Object.keys(counts?.labels ?? {}), "uncertain"]));
  const decisions: Decision[] = ["ACCEPT", "REJECT", "REVIEW"];

  return (
    <div className="panel stats">
      <div className="total">
        <div className="total-number">{total}</div>
        <div className="muted">tomatoes inspected</div>
      </div>

      <div className="eyebrow">Ripeness</div>
      <div className="breakdown">
        {labels.map((l) => {
          const n = counts?.labels[l] ?? 0;
          const color = CATEGORY_COLOR[categoryOf(l)] ?? "var(--uncertain)";
          return (
            <div key={l} className="breakdown-row">
              <span className="dot" style={{ background: color }} />
              <span className="grow">{capitalize(l)}</span>
              <strong>{n}</strong>
              <span className="muted pct">{total ? pct(n / total) : "—"}</span>
              <div className="bar">
                <div style={{ width: total ? pct(n / total) : "0%", background: color }} />
              </div>
            </div>
          );
        })}
      </div>

      <div className="eyebrow">Sorting</div>
      <div className="bins">
        {decisions.map((d) => (
          <div key={d} className="bin" style={{ borderColor: DECISION_COLOR[d] }}>
            <div className="bin-count">{counts?.decisions[d] ?? 0}</div>
            <div className="bin-label" style={{ color: DECISION_COLOR[d] }}>
              {d}
            </div>
          </div>
        ))}
      </div>

      <div className="kpis">
        <Kpi label="per minute" value={status?.throughput_per_min ?? 0} />
        <Kpi label="fps" value={status?.fps.toFixed(1) ?? "0"} />
        <Kpi label="inference" value={`${status?.inference_ms.toFixed(0) ?? 0} ms`} />
        <Kpi label="on belt" value={status?.active_tracks ?? 0} />
      </div>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="kpi">
      <div className="kpi-value">{value}</div>
      <div className="muted small">{label}</div>
    </div>
  );
}
