import { useEffect, useRef, useState } from "react";
import type { InspectionEvent, PipelineStatus } from "./types";

const MAX_EVENTS = 200;

interface Telemetry {
  connected: boolean;
  status: PipelineStatus | null;
  model: { installed: boolean; error: string | null } | null;
  events: InspectionEvent[];
}

/** Subscribes to the backend /ws feed and keeps a rolling list of inspection events. */
export function useTelemetry(): Telemetry {
  const [state, setState] = useState<Telemetry>({ connected: false, status: null, model: null, events: [] });
  const sessionRef = useRef<number | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: number | undefined;
    let closed = false;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => setState((s) => ({ ...s, connected: true }));
      ws.onclose = () => {
        setState((s) => ({ ...s, connected: false }));
        if (!closed) retry = window.setTimeout(connect, 1500);
      };
      ws.onmessage = (msg) => {
        const data = JSON.parse(msg.data);
        if (data.type !== "status") return;
        const status: PipelineStatus = data.status;
        setState((s) => {
          let events = s.events;
          // new session or counters reset → start a fresh event list
          if (status.session_id !== sessionRef.current || status.counts.total < (s.status?.counts.total ?? 0)) {
            events = [];
            sessionRef.current = status.session_id;
          }
          if (data.events.length) {
            const known = new Set(events.map((e) => e.seq));
            const fresh = (data.events as InspectionEvent[]).filter((e) => !known.has(e.seq)).reverse();
            events = [...fresh, ...events].slice(0, MAX_EVENTS);
          }
          return { connected: true, status, model: data.model, events };
        });
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry);
      ws?.close();
    };
  }, []);

  return state;
}
