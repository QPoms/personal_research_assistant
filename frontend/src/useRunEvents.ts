import { useEffect, useRef, useState } from "react";

import type { RunEvent, StreamState } from "./types";

interface RunEventStream {
  events: RunEvent[];
  state: StreamState;
}

export function useRunEvents(runId: string | null): RunEventStream {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [state, setState] = useState<StreamState>("closed");
  const cursor = useRef(0);

  useEffect(() => {
    cursor.current = 0;
    setEvents([]);
    if (!runId) {
      setState("closed");
      return;
    }

    setState("connecting");
    const url = `/api/v1/runs/${encodeURIComponent(runId)}/events?after_seq=${cursor.current}`;
    const source = new EventSource(url);

    source.onopen = () => setState("live");
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as RunEvent;
      if (event.seq <= cursor.current) return;
      cursor.current = event.seq;
      setEvents((current) => [...current, event]);
    };

    const eventTypes = [
      "run.created",
      "run.stage_changed",
      "run.completed",
      "run.failed",
      "run.interrupted",
      "run.cancelled",
      "plan.created",
      "task.started",
      "task.completed",
      "model.started",
      "model.retrying",
      "model.completed",
      "tool.started",
      "tool.completed",
      "tool.failed",
      "evidence.selected",
      "context.compacted",
      "report.updated",
    ];
    const receive = (message: MessageEvent<string>) => {
      const event = JSON.parse(message.data) as RunEvent;
      if (event.seq <= cursor.current) return;
      cursor.current = event.seq;
      setEvents((current) => [...current, event]);
      if (["completed", "failed", "interrupted", "cancelled"].includes(event.status ?? "")) {
        source.close();
        setState("closed");
      }
    };
    eventTypes.forEach((type) => source.addEventListener(type, receive as EventListener));

    source.onerror = () => {
      setState(source.readyState === EventSource.CLOSED ? "closed" : "error");
    };

    return () => source.close();
  }, [runId]);

  return { events, state };
}
