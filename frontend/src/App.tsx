import { useCallback, useEffect, useMemo, useState } from "react";

import { getRun, listRuns } from "./api";
import type { RunEvent, RunRecord, RunStatus, StreamState } from "./types";
import { useRunEvents } from "./useRunEvents";
import "./styles.css";

const STATUS_LABELS: Record<RunStatus, string> = {
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  interrupted: "已中断",
  cancelled: "已取消",
};

const STAGE_LABELS: Record<string, string> = {
  building_agent: "构建 Agent",
  running_agent: "执行研究",
  validating_report: "校验报告",
  completed: "完成",
};

const EVENT_LABELS: Record<string, string> = {
  "run.created": "创建研究运行",
  "run.stage_changed": "研究阶段推进",
  "run.completed": "研究运行完成",
  "run.failed": "研究运行失败",
  "run.interrupted": "研究运行中断",
  "run.cancelled": "研究运行取消",
  "plan.created": "生成研究计划",
  "task.started": "开始子任务",
  "task.completed": "完成子任务",
  "model.started": "发起模型调用",
  "model.retrying": "模型调用重试",
  "model.completed": "模型调用完成",
  "tool.started": "调用研究工具",
  "tool.completed": "研究工具完成",
  "tool.failed": "研究工具失败",
  "evidence.selected": "选择有效证据",
  "context.compacted": "压缩研究上下文",
  "report.updated": "更新研究报告",
};

const STREAM_LABELS: Record<StreamState, string> = {
  connecting: "连接中",
  live: "实时同步",
  closed: "历史已同步",
  error: "正在重连",
};

function formatTime(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function sourceLabel(value: string): string {
  if (value === "mixed") return "混合研究";
  if (value === "personal") return "私人知识库";
  return "联网研究";
}

function EventCard({ event }: { event: RunEvent }) {
  const hasPayload = Object.keys(event.payload).length > 0;
  return (
    <li className={`event event--${event.event_type.split(".").at(-1)}`}>
      <div className="event__rail" aria-hidden="true">
        <span className="event__dot" />
      </div>
      <article className="event__card">
        <div className="event__heading">
          <div>
            <span className="event__sequence">#{event.seq}</span>
            <h3>{EVENT_LABELS[event.event_type] ?? event.event_type}</h3>
          </div>
          <time dateTime={event.occurred_at}>{formatTime(event.occurred_at)}</time>
        </div>
        <div className="event__meta">
          <span>{event.source}</span>
          {event.agent && <span>Agent · {event.agent}</span>}
          {event.task_id && <span>任务 · {event.task_id}</span>}
          {event.stage && <span>{STAGE_LABELS[event.stage] ?? event.stage}</span>}
        </div>
        {hasPayload && (
          <details>
            <summary>查看事件数据</summary>
            <pre>{JSON.stringify(event.payload, null, 2)}</pre>
          </details>
        )}
      </article>
    </li>
  );
}

function App() {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<RunRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { events, state: streamState } = useRunEvents(selectedId);

  const refreshRuns = useCallback(async () => {
    setError(null);
    try {
      const nextRuns = await listRuns();
      setRuns(nextRuns);
      setSelectedId((current) => current ?? nextRuns[0]?.run_id ?? null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法加载运行记录");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshRuns();
  }, [refreshRuns]);

  useEffect(() => {
    if (!selectedId) {
      setSelected(null);
      return;
    }
    void getRun(selectedId)
      .then(setSelected)
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "无法加载运行详情");
      });
  }, [selectedId, events.length]);

  const lastEvent = events.at(-1);
  const displayStatus = (lastEvent?.status as RunStatus | null) ?? selected?.status;
  const displayStage = lastEvent?.stage ?? selected?.stage;
  const summary = useMemo(
    () => ({
      tools: events.filter((event) => event.event_type.startsWith("tool.")).length,
      retries: events.filter((event) => event.event_type === "model.retrying").length,
      evidence: events.filter((event) => event.event_type === "evidence.selected").length,
    }),
    [events],
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">P</div>
        <div>
          <p>Personal Research Assistant</p>
          <span>研究运行控制台</span>
        </div>
        <div className="topbar__spacer" />
        <span className="system-state"><i /> 本地运行账本</span>
      </header>

      <aside className="sidebar">
        <div className="sidebar__heading">
          <div>
            <span className="eyebrow">RUN LEDGER</span>
            <h1>研究记录</h1>
          </div>
          <button type="button" onClick={() => void refreshRuns()} aria-label="刷新运行列表">
            ↻
          </button>
        </div>
        <nav aria-label="研究运行列表">
          {loading && <p className="quiet">正在读取运行账本…</p>}
          {!loading && runs.length === 0 && <p className="quiet">尚无研究运行。</p>}
          {runs.map((run) => (
            <button
              type="button"
              className={`run-card ${selectedId === run.run_id ? "run-card--active" : ""}`}
              key={run.run_id}
              onClick={() => setSelectedId(run.run_id)}
            >
              <span className={`status-dot status-dot--${run.status}`} />
              <span className="run-card__content">
                <strong>{run.query}</strong>
                <span>{sourceLabel(run.source_mode)} · {formatTime(run.started_at)}</span>
              </span>
              <small>{STATUS_LABELS[run.status]}</small>
            </button>
          ))}
        </nav>
      </aside>

      <main className="workspace">
        {error && <div className="error-banner" role="alert">{error}</div>}
        {!selected ? (
          <section className="empty-state">
            <span>⌁</span>
            <h2>选择一条研究记录</h2>
            <p>这里会按持久化序号回放 Chief、Researcher、模型和工具事件。</p>
          </section>
        ) : (
          <>
            <section className="run-hero">
              <div className="run-hero__title">
                <span className="eyebrow">RUN · {selected.run_id}</span>
                <h2>{selected.query}</h2>
                <div className="run-hero__tags">
                  <span>{sourceLabel(selected.source_mode)}</span>
                  <span>{STAGE_LABELS[displayStage ?? ""] ?? displayStage}</span>
                  {displayStatus && (
                    <span className={`status-pill status-pill--${displayStatus}`}>
                      {STATUS_LABELS[displayStatus]}
                    </span>
                  )}
                </div>
              </div>
              <div className={`stream-state stream-state--${streamState}`} aria-live="polite">
                <i /> {STREAM_LABELS[streamState]}
              </div>
            </section>

            <section className="metrics" aria-label="运行统计">
              <div><span>事件</span><strong>{events.length}</strong></div>
              <div><span>工具调用</span><strong>{summary.tools}</strong></div>
              <div><span>模型重试</span><strong>{summary.retries}</strong></div>
              <div><span>证据选择</span><strong>{summary.evidence}</strong></div>
              <div><span>Web 成本</span><strong>${selected.web_research_cost.toFixed(4)}</strong></div>
            </section>

            <section className="timeline-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">EVENT STREAM</span>
                  <h2>执行时间线</h2>
                </div>
                <span>最后序号 · {lastEvent?.seq ?? 0}</span>
              </div>
              {events.length === 0 ? (
                <p className="quiet">正在等待首个运行事件…</p>
              ) : (
                <ol className="timeline">
                  {events.map((event) => <EventCard event={event} key={event.event_id} />)}
                </ol>
              )}
            </section>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
