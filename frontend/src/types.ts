export type RunStatus =
  | "running"
  | "completed"
  | "failed"
  | "interrupted"
  | "cancelled";

export type RunStage =
  | "building_agent"
  | "running_agent"
  | "validating_report"
  | "completed";

export interface RunRecord {
  schema_version: "run-record/v1";
  run_id: string;
  query: string;
  source_mode: string;
  status: RunStatus;
  stage: RunStage;
  started_at: string;
  finished_at: string | null;
  web_research_cost: number;
  report_path: string;
  guidelines: string[];
  error: string | null;
}

export interface RunEvent {
  schema_version: "run-event/v1";
  event_id: string;
  seq: number;
  run_id: string;
  event_type: string;
  occurred_at: string;
  source: string;
  status: string | null;
  stage: string | null;
  agent: string | null;
  task_id: string | null;
  payload: Record<string, unknown>;
}

export type StreamState = "connecting" | "live" | "closed" | "error";
