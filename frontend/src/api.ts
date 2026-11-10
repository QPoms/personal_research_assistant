import type { RunRecord } from "./types";

async function requestJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`请求失败：HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export function listRuns(): Promise<RunRecord[]> {
  return requestJson<RunRecord[]>("/api/v1/runs?limit=50");
}

export function getRun(runId: string): Promise<RunRecord> {
  return requestJson<RunRecord>(`/api/v1/runs/${encodeURIComponent(runId)}`);
}
