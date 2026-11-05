import { DECISION_COLOR, capitalize, clockTime, colorOf } from "../format";
import type { InspectionEvent } from "../types";

export function EventLog({ events, sessionId }: { events: InspectionEvent[]; sessionId: number | null }) {
  return (
    <div className="panel log">
      <div className="panel-head">
        <div className="eyebrow">Inspection log</div>
        <a className="btn ghost small" href={`/api/inspections/export.csv${sessionId ? `?session_id=${sessionId}` : ""}`}>
          Export CSV
        </a>
      </div>
      {events.length === 0 ? (
        <div className="muted">No inspections in this session yet.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Time</th>
                <th>Track</th>
                <th>Grade</th>
                <th>Decision</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.seq} title={e.reason || undefined}>
                  <td className="mono">{e.number}</td>
                  <td className="mono muted">{clockTime(e.created_at)}</td>
                  <td className="mono">#{e.track_id}</td>
                  <td>
                    <span className="dot" style={{ background: colorOf(e.label) }} /> {capitalize(e.label)}
                  </td>
                  <td>
                    <span className="decision-chip" style={{ background: DECISION_COLOR[e.decision] }}>
                      {e.decision}
                    </span>
                  </td>
                  <td className="event-reason">{e.reason || "Grading checks passed"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="note small">Each track is counted once using the selected counting trigger. Uncertain means there was not enough evidence under the selected grading mode.</p>
    </div>
  );
}
