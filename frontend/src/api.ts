import type { ModelStatus, PipelineStatus, Settings, SourceType } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

export const api = {
  model: () => request<ModelStatus>("/api/model"),
  reloadModel: () => post<ModelStatus>("/api/model/reload"),
  settings: () => request<Settings>("/api/settings"),
  patchSettings: (patch: Partial<Settings> | Record<string, unknown>) =>
    request<Settings>("/api/settings", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ upload_id: string; name: string; duration_s: number | null }>("/api/upload", {
      method: "POST",
      body: form,
    });
  },
  start: (type: SourceType, value: string) => post<PipelineStatus>("/api/session/start", { type, value }),
  pause: () => post<PipelineStatus>("/api/session/pause"),
  resume: () => post<PipelineStatus>("/api/session/resume"),
  stop: () => post<PipelineStatus>("/api/session/stop"),
  reset: () => post<PipelineStatus>("/api/session/reset"),
};
