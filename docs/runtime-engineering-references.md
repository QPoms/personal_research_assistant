# Runtime engineering references

This project keeps its research semantics (Deep Agents Chief/Researcher, GPT
Researcher for Web, and embedded Personal Knowledge research) while borrowing
small, well-defined runtime patterns from mature open-source systems. It does
not import another project's complete platform or adopt multi-tenant services.

## Reference map

| Need | Primary reference | Pattern retained | Pattern intentionally deferred |
|---|---|---|---|
| Durable graph state and resume | LangGraph persistence and interrupts | Checkpoint boundaries, thread/run identity, replayable state | Hosted LangGraph platform and Postgres |
| Agent runtime events | OpenHands EventStream | Typed events, Action/Observation correlation, append-only history | Sandbox runtime and recursive coding-agent delegation |
| Workflow execution layers | Dify workflow engine | Separate domain events, persistence projection, command/cancel channel | Celery, Redis Pub/Sub, tenancy and plugin platform |
| Context tiers | Letta memory architecture | Small in-context state plus out-of-context, on-demand artifacts | User/persona memory and autonomous long-term learning |
| LLM observability and evaluation | Arize Phoenix/OpenInference | OpenTelemetry-compatible traces and later retrieval evaluations | Bundling an observability server into the core runtime |
| Lightweight research UI | GPT Researcher | Research progress presentation and basic streaming UI | Its connection-scoped WebSocket protocol, which has no durable replay cursor |

## Decisions

1. Runtime events are application-owned and provider-independent.
2. Every event has a schema version, immutable event ID, run ID, UTC timestamp,
   event type, source, and per-run monotonic sequence.
3. `run.json` remains a convenient current-state projection; `events.jsonl` is
   the append-only audit and replay stream.
4. Long model/tool payloads stay in artifacts and evidence storage. Events carry
   identifiers and bounded metadata rather than entire documents.
5. JSONL is the first single-process implementation. SQLite becomes the next
   persistence adapter before background workers or an HTTP API are introduced.
6. Tool execution is not blindly retried at the event layer. Idempotency and
   retry boundaries must be defined per tool.
7. Authentication, workspaces, Redis, Celery, Docker and multi-tenancy remain
   out of scope until local single-user reliability requires them.

## Source links

- LangGraph persistence: <https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/persistence.mdx>
- LangGraph interrupts: <https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/interrupts.mdx>
- OpenHands backend architecture: <https://github.com/OpenHands/docs/blob/main/openhands/usage/architecture/backend.mdx>
- Dify workflow runner: <https://github.com/langgenius/dify/blob/main/api/core/app/apps/workflow/app_runner.py>
- Letta memory architecture: <https://github.com/letta-ai/skills/blob/main/letta/agent-development/references/memory-architecture.md>
- Arize Phoenix: <https://github.com/Arize-ai/phoenix>
