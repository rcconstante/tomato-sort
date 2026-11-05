import { useEffect, useRef, useState } from "react";
import type { InspectionLine, PipelineStatus } from "../types";

interface Props {
  status: PipelineStatus | null;
  line: InspectionLine | null;
  editing: boolean;
  onLineCommit: (line: InspectionLine) => void;
}

const STATE_LABEL: Record<string, string> = {
  running: "LIVE",
  paused: "PAUSED",
  finished: "FINISHED",
  stopped: "STOPPED",
  error: "ERROR",
};

export function VideoPanel({ status, line, editing, onLineCommit }: Props) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState<InspectionLine | null>(line);
  const dragging = useRef<null | 1 | 2>(null);

  useEffect(() => {
    if (!dragging.current) setDraft(line);
  }, [line]);

  const state = status?.state ?? "idle";
  const hasStream = status?.session_id != null && state !== "idle";
  const w = status?.source.width || 16;
  const h = status?.source.height || 9;

  const toNorm = (e: React.PointerEvent) => {
    const r = boxRef.current!.getBoundingClientRect();
    const clamp = (v: number) => Math.min(1, Math.max(0, v));
    return { x: clamp((e.clientX - r.left) / r.width), y: clamp((e.clientY - r.top) / r.height) };
  };

  const onMove = (e: React.PointerEvent) => {
    if (!dragging.current || !draft) return;
    const p = toNorm(e);
    setDraft(dragging.current === 1 ? { ...draft, x1: p.x, y1: p.y } : { ...draft, x2: p.x, y2: p.y });
  };

  const onUp = () => {
    if (dragging.current && draft) onLineCommit(draft);
    dragging.current = null;
  };

  return (
    <div className="video-frame" style={{ aspectRatio: `${w} / ${h}` }} ref={boxRef} onPointerMove={onMove} onPointerUp={onUp}>
      {hasStream ? (
        <img className="video" src={`/api/stream.mjpg?session=${status!.session_id}`} alt="Annotated conveyor stream" draggable={false} />
      ) : (
        <div className="video-empty">
          <div className="video-empty-title">No active source</div>
          <div>Choose a video, camera or stream below and press Start.</div>
        </div>
      )}

      {hasStream && (
        <div className={`state-pill state-${state}`}>
          <span className="dot" />
          {STATE_LABEL[state] ?? state.toUpperCase()}
          {state === "running" && <span className="muted">· {status!.fps.toFixed(0)} fps</span>}
        </div>
      )}

      {editing && draft && (
        <>
          <svg className="line-editor" viewBox="0 0 1 1" preserveAspectRatio="none">
            <line x1={draft.x1} y1={draft.y1} x2={draft.x2} y2={draft.y2} />
          </svg>
          {([1, 2] as const).map((i) => (
            <div
              key={i}
              className="line-handle"
              style={{ left: `${(i === 1 ? draft.x1 : draft.x2) * 100}%`, top: `${(i === 1 ? draft.y1 : draft.y2) * 100}%` }}
              onPointerDown={(e) => {
                dragging.current = i;
                boxRef.current?.setPointerCapture(e.pointerId);
              }}
            />
          ))}
          <div className="line-hint">Drag the handles to place the inspection line</div>
        </>
      )}
    </div>
  );
}
