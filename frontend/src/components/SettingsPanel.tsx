import { capitalize } from "../format";
import type { Decision, Settings } from "../types";

interface Props {
  settings: Settings;
  classes: string[];
  editingLine: boolean;
  onToggleEditLine: () => void;
  onChange: (patch: Record<string, unknown>) => void;
}

const DECISIONS: Decision[] = ["ACCEPT", "REJECT", "REVIEW"];

export function SettingsPanel({ settings, classes, editingLine, onToggleEditLine, onChange }: Props) {
  const sortKeys = Array.from(new Set([...classes.map((c) => c.toLowerCase()), ...Object.keys(settings.sorting), "uncertain"]));

  return (
    <div className="panel settings">
      <div className="eyebrow">Inspection line</div>
      <div className="row">
        <button className={editingLine ? "btn primary" : "btn"} onClick={onToggleEditLine}>
          {editingLine ? "Done" : "Edit line"}
        </button>
        <label className="field-inline">
          Count direction
          <select value={settings.line.direction} onChange={(e) => onChange({ line: { direction: e.target.value } })}>
            <option value="forward">Forward →</option>
            <option value="reverse">← Reverse</option>
            <option value="both">Both ↔</option>
          </select>
        </label>
      </div>

      <div className="eyebrow">Detection & grading</div>
      <Slider label="Detector confidence" value={settings.detect_conf} min={0.05} max={0.9} step={0.05} fmt={(v) => v.toFixed(2)} onChange={(v) => onChange({ detect_conf: v })} />
      <Slider label="Certain above" value={settings.certain_conf} min={0.1} max={0.95} step={0.05} fmt={(v) => v.toFixed(2)} onChange={(v) => onChange({ certain_conf: v })} />
      <Slider label="Min. frames per tomato" value={settings.min_observations} min={1} max={30} step={1} fmt={(v) => String(v)} onChange={(v) => onChange({ min_observations: v })} />
      <Slider label="Required vote agreement" value={settings.min_vote_share} min={0.5} max={1} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`} onChange={(v) => onChange({ min_vote_share: v })} />

      <div className="eyebrow">Sorting rules</div>
      <div className="rules">
        {sortKeys.map((k) => (
          <label key={k} className="field-inline">
            {capitalize(k)}
            <select value={settings.sorting[k] ?? "REVIEW"} onChange={(e) => onChange({ sorting: { [k]: e.target.value } })}>
              {DECISIONS.map((d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
          </label>
        ))}
      </div>

      <div className="eyebrow">Display & playback</div>
      <div className="toggles">
        <Toggle label="Bounding boxes" checked={settings.show_boxes} onChange={(v) => onChange({ show_boxes: v })} />
        <Toggle label="Inspection line" checked={settings.show_line} onChange={(v) => onChange({ show_line: v })} />
        <Toggle label="Loop video files" checked={settings.loop_video} onChange={(v) => onChange({ loop_video: v })} />
        <Toggle label="Real-time playback" checked={settings.realtime_playback} onChange={(v) => onChange({ realtime_playback: v })} />
      </div>
    </div>
  );
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; fmt: (v: number) => string; onChange: (v: number) => void }) {
  return (
    <label className="slider">
      <span className="slider-head">
        <span>{props.label}</span>
        <span className="mono">{props.fmt(props.value)}</span>
      </span>
      <input type="range" min={props.min} max={props.max} step={props.step} value={props.value} onChange={(e) => props.onChange(Number(e.target.value))} />
    </label>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch" />
      {label}
    </label>
  );
}
