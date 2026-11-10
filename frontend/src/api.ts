import type { RunRecord } from "./types";

interface RunOperationResponse {
  run_id: string;
  accepted: boolean;
  operation: "start" | "resume" | "cancel";
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { Accept: "application/json", "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `请求失败：HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export function listRuns(): Promise<RunRecord[]> {
  return requestJson<RunRecord[]>("/api/v1/runs?limit=50");
}

export function getRun(runId: string): Promise<RunRecord> {
  return requestJson<RunRecord>(`/api/v1/runs/${encodeURIComponent(runId)}`);
}

export function resumeRun(runId: string): Promise<RunOperationResponse> {
  return requestJson<RunOperationResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/resume`,
    { method: "POST" },
  );
}

export function cancelRun(runId: string): Promise<RunOperationResponse> {
  return requestJson<RunOperationResponse>(
    `/api/v1/runs/${encodeURIComponent(runId)}/cancel`,
    { method: "POST", body: JSON.stringify({ reason: "用户从运行面板请求取消。" }) },
  );
}
