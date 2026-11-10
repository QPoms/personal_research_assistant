# Personal Research Assistant

Personal Research Assistant coordinates two independent research sources at
the Agent layer:

- **GPT Researcher 0.16.0** performs Web research through its public Python API.
- **Personal Knowledge** is a slim, embedded research engine migrated from
  SpecRAG: bounded task planning, task Researchers, Release-aware BM25/Dense
  retrieval, RRF, source validation, cross-task evidence merging, and a
  grounded local brief for the Chief Editor.
- **Deep Agents 0.7.8** supplies the Chief Editor harness, Researcher subagent,
  Todo management, filesystem tools, and task delegation.

The project does not reimplement GPT Researcher's engine or run SpecRAG as a
separate HTTP service. SpecRAG's frontend, Docker deployment, authentication,
workspaces, multi-user permissions, memory, context compression, API server,
and worker lifecycle are intentionally excluded.

## Repository layout

The repository follows GPT Researcher's monorepo principle: each top-level
directory represents a real runtime or product boundary. It does not use a
generic `src/` container.

```text
backend/                    # application entry point, settings, runs, outputs
  api/                      # FastAPI run control, snapshots and replayable SSE
  research/                 # run orchestration and report validation
  runtime/                  # model construction, checkpoints, worker supervisor
  storage/                  # isolated run directories and SQLite runtime ledger
deep_agents/                # Chief Editor and Researcher team construction
frontend/                   # local React runtime console
personal_knowledge/         # embedded private-knowledge retrieval engine
  infrastructure/           # optional local vector-store adapter
  retrieval/                # corpus, BM25, dense, RRF, clauses, snapshots
  research/                 # Planner, task Researchers, merger, local brief
research_tools/             # Agent-facing Web/personal tools and policies
docs/                       # architecture decisions
tests/                      # unit and contract tests
cli.py                      # repository-level development entry point
```

There is currently no `memory/`, `mcp-server/`, or `terraform/` directory
because the project does not implement those capabilities. New top-level
boundaries should be added only when executable code actually exists.
See [docs/architecture.md](docs/architecture.md) for the dependency rules and
the comparison with the GPT Researcher reference repository.

## Runtime flow

```text
User query
    -> backend
        -> Chief Editor (scope, plan, delegate, review, assemble)
            -> Researcher subagents
                -> GPT Researcher [Web]
                -> Personal Knowledge Tool [private corpus]
                    -> Local Planner
                    -> bounded task Researchers
                    -> evidence merger
                    -> grounded local brief
        -> outputs/run_<id>/sections/*.md
        -> outputs/run_<id>/report.md
```

Source authorization is enforced in code when the agents are constructed:

| Mode | Chief tools | Researcher tools |
|---|---|---|
| `web` | `quick_search` | `quick_search`, `deep_research` |
| `personal` | none; delegates local work | `search_personal_knowledge` |
| `mixed` | `quick_search` | all three tools |

## Requirements

- Python 3.12+
- `uv`
- Model and search-provider credentials supported by GPT Researcher 0.16.0
- For `personal` or `mixed`: a local Markdown corpus and compatible index

The repository pins top-level integrations in `pyproject.toml` and the complete
resolved graph in `uv.lock`. GPT Researcher is pinned to an audited upstream
commit because the PyPI 0.16.0 artifact currently fails during import.

## Setup

```powershell
uv sync --extra personal --extra test
Copy-Item .env.example .env
```

The `personal` extra installs dense retrieval. Without it, the Personal
Knowledge Tool remains usable through its verified BM25/RRF fallback and marks
the result partial when the dense channel is unavailable.

Configure `.env`. `RETRIEVER=duckduckgo` works without a separate search key
for development; Tavily requires its own key. For private knowledge, set:

For OpenAI-compatible third-party gateways, keep
`DEEP_AGENT_USE_RESPONSES_API=false` unless the gateway implements the OpenAI
Responses API correctly. `DEEP_AGENT_MAX_OUTPUT_TOKENS` bounds each Chief
Editor/Researcher model response. The repository-local `.env` is authoritative
at runtime so stale provider credentials inherited from the launching shell do
not silently replace the project's configured key.

Model calls use one application-owned retry budget shared by Deep Agents and
embedded Personal Knowledge research. The default is five retries after the
initial request, with exponential backoff and jitter. Only transient failures
(429, 5xx, connection and timeout errors) are retried; permanent authentication,
permission, model-name, and request errors fail immediately. Provider-SDK
retries are disabled so retry layers cannot multiply attempts. Empty or malformed
JSON responses from the local Planner/Researcher/Synthesizer consume the same
five-retry budget. This policy does not change request timeout configuration.

```env
PERSONAL_KNOWLEDGE_ROOT=personal_knowledge_data
PERSONAL_KNOWLEDGE_CORPUS_DIR=data/3gpp
PERSONAL_KNOWLEDGE_INDEX_DIR=.local/indexes/specifications
PERSONAL_KNOWLEDGE_RELEASE=17
PERSONAL_KNOWLEDGE_RESEARCH_ENABLED=true
PERSONAL_KNOWLEDGE_ALLOW_EXTERNAL_MODEL=false
```

These paths and the Release are backend configuration, never model-controlled
Tool arguments. Existing SpecRAG data can be reused by pointing
`PERSONAL_KNOWLEDGE_ROOT` at its data root; no SpecRAG service needs to run.
Full local research sends selected evidence excerpts to the configured
Planner/Researcher/Synthesizer model. The privacy gate must be explicitly set
to `true` before that disclosure is allowed. See
[docs/personal-knowledge-workflow.md](docs/personal-knowledge-workflow.md).

## Run

```powershell
uv run research-assistant `
  --query "分析近年来 RAG 技术的发展现状" `
  --source-mode web
```

During repository development, the equivalent entry point is:

```powershell
uv run python cli.py --query "..." --source-mode mixed
```

Each run gets an isolated directory under `outputs/` containing `run.json`,
an append-only `events.jsonl`, `agent-trace.json`, section drafts, and
`report.md`. Runtime events use the versioned `run-event/v1` contract and a
strictly increasing per-run sequence number so future API and frontend clients
can replay from a known cursor. The canonical run snapshot and event stream are
committed atomically to `outputs/runtime.sqlite3`; the JSON files are readable
projections rebuilt from that database. Relative output paths are
anchored to the repository root. Inside Deep Agents, the same run directory is
exposed only as virtual `/`; drafts use `/sections/...` and the final report
uses `/report.md`.

Deep Agents graph state is checkpointed synchronously to
`outputs/checkpoints.sqlite3`, using the run ID as the LangGraph thread ID. If
an OS/process interruption leaves a resumable run, continue from the latest
completed graph superstep instead of restarting its research plan:

```powershell
uv run research-assistant --resume-run <run-id>
```

User-requested cancellation and failed/completed runs are terminal and cannot
be resumed. Checkpoint deserialization runs in strict MsgPack mode without
pickle fallback.

Callers may pass a run-scoped `CancellationToken` to `run_research`. A requested
cancel stops the active Deep Agents task, propagates through GPT Researcher and
the embedded SpecRAG pipeline, interrupts local model retry backoff, and commits
the terminal `cancelled` snapshot together with a `run.cancelled` event.

Start the local API with `uv run research-assistant-api`. Run snapshots are
available at `GET /api/v1/runs/{run_id}`. New runs are accepted by
`POST /api/v1/runs`; active runs can be cooperatively cancelled with
`POST /api/v1/runs/{run_id}/cancel`, and interrupted runs resume from their
latest durable checkpoint with `POST /api/v1/runs/{run_id}/resume`. The event endpoint
`GET /api/v1/runs/{run_id}/events` first replays SQLite history and then tails
new events. Clients resume with `after_seq` or the SSE `Last-Event-ID` header;
the durable per-run sequence is used as the SSE event ID.

The API owns a lightweight in-process worker supervisor rather than Celery or
Redis. Every active run has a SQLite worker lease and heartbeat. On graceful
shutdown, active tasks are persisted as interrupted; on restart, the Watchdog
marks expired leases interrupted so the same run can be resumed. Heartbeats
are operational liveness metadata and do not change model or research request
timeouts.

The minimal runtime console lives in `frontend/`. Build it once, then the same
FastAPI process serves the production assets and API on port 8000:

```powershell
Set-Location frontend
npm install
npm run build
Set-Location ..
uv run research-assistant-api
```

For frontend development, `npm run dev` starts Vite on port 5173 and proxies
`/api` to the local FastAPI process. The console intentionally contains no
login, workspace, multi-tenant, or cloud deployment layer.

## Tests

```powershell
uv run pytest
```

The suite covers source authorization, evidence normalization, upstream Web
engine contracts, embedded retrieval and source verification, prompts,
run-path isolation, checkpoints, worker leases, cancellation, Watchdog recovery,
and HTTP/SSE runtime control. Live Web acceptance requires provider credentials.

## Upstream reference

`gpt-researcher-main/` is a local-only source reference and is excluded by
`.gitignore`. It is not a runtime dependency and is retained locally for the
ongoing implementation audit. Provenance is recorded in
`THIRD_PARTY_NOTICES.md`.
