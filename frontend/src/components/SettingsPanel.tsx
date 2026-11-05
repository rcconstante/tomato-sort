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
      <div className="eyebrow">Counting</div>
      <label className="field-inline">
        Log tomatoes
        <select value={settings.count_trigger} onChange={(e) => onChange({ count_trigger: e.target.value })}>
          <option value="detection">On confirmed detection</option>
          <option value="line">On inspection-line crossing</option>
        </select>
      </label>
      <p className="note small">{settings.count_trigger === "detection"
        ? "Counts each tracked tomato as soon as enough observations are available, anywhere in the frame. The inspection line is not used."
        : "Counts only tracks observed crossing the inspection line in the selected direction."}</p>
      {settings.count_trigger === "line" && <>
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
      </>}

      <div className="eyebrow">Detection & grading</div>
      <label className="field-inline">
        Counting mode
        <select value={settings.grading_mode} onChange={(e) => onChange({ grading_mode: e.target.value })}>
          <option value="binary">Simple ripe / unripe</option>
          <option value="strict">Strict quality review</option>
        </select>
      </label>
      <p className="note small">Simple mode uses the winning tracked class after enough observations. Strict mode also requires confidence and agreement checks.</p>
      <label className="field-inline">
        Processing detail
        <select value={settings.imgsz} onChange={(e) => onChange({ imgsz: Number(e.target.value) })}>
          <option value={512}>Fast (512)</option>
          <option value={640}>Detailed (640)</option>
          <option value={960}>High detail (960)</option>
        </select>
      </label>
      <details className="advanced-settings">
      <summary>Advanced detection settings</summary>
      <Slider label="Detector confidence" value={settings.detect_conf} min={0.05} max={0.9} step={0.05} fmt={(v) => v.toFixed(2)} onChange={(v) => onChange({ detect_conf: v })} />
      <Slider label="Strict-mode minimum confidence" value={settings.certain_conf} min={0.1} max={0.95} step={0.05} fmt={(v) => v.toFixed(2)} onChange={(v) => onChange({ certain_conf: v })} />
      <Slider label="Min. frames per tomato" value={settings.min_observations} min={1} max={30} step={1} fmt={(v) => String(v)} onChange={(v) => onChange({ min_observations: v })} />
      <Slider label="Strict-mode agreement" value={settings.min_vote_share} min={0.5} max={1} step={0.05} fmt={(v) => v.toFixed(2)} onChange={(v) => onChange({ min_vote_share: v })} />
      </details>

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
      <label className="field-inline">
        Object highlighting
        <select value={settings.outline_mode} onChange={(e) => onChange({ outline_mode: e.target.value })}>
          <option value="contour">Estimated contour</option>
          <option value="box">Bounding box</option>
        </select>
      </label>
      <p className="note small">Contours estimate visible color boundaries inside detections. Boxes are used when a boundary cannot be estimated. Highlighting does not change grading.</p>
      <div className="toggles">
        <Toggle label="Object highlights" checked={settings.show_boxes} onChange={(v) => onChange({ show_boxes: v })} />
        {settings.count_trigger === "line" && <Toggle label="Inspection line" checked={settings.show_line} onChange={(v) => onChange({ show_line: v })} />}
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
