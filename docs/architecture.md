# Repository architecture

## Why the repository resembles GPT Researcher

GPT Researcher is organized as a monorepo with several different kinds of
top-level boundary:

| Reference directory | Actual role |
|---|---|
| `gpt_researcher/` | Reusable research engine: agent, actions, retrievers, scraping, context selection, vector stores, and provider adapters. |
| `backend/` | Product-facing FastAPI/WebSocket server, report storage, chat, file handling, and static frontend hosting. |
| `deep_agents/` | A separate Deep Agents harness and benchmark implementation that imports the core engine. |
| `frontend/` | Static and Next.js clients that communicate with the backend. |
| `multi_agents/` | Alternative orchestration implementation, not the core engine. |
| `evals/` | Evaluation workloads and metrics. |
| `mcp-server/` | Optional integration boundary. |
| `terraform/` | Deployment infrastructure. |

The important principle is not the number of folders. Each directory owns a
real capability and has a clear dependency direction. This project applies the
same principle at its current, smaller scope.

## Current boundaries

```text
backend
  -> deep_agents
  -> research_tools

deep_agents
  -> research_tools
  -> backend.config/runtime (application composition)

research_tools
  -> backend.config
  -> GPT Researcher dependency
  -> personal_knowledge

personal_knowledge
  -> no backend or agent dependency
```

Responsibilities:

- `backend/` owns application configuration, command-line/API composition,
  run lifecycle, output paths, and report validation.
- `deep_agents/` owns Chief Editor and Researcher prompts and construction. It
  uses Deep Agents' native Todo, filesystem, and subagent mechanisms.
- `research_tools/` is the anti-corruption layer around external engines. It
  enforces `web`, `personal`, and `mixed` authorization and normalizes evidence.
- `personal_knowledge/retrieval/` is a reusable, framework-independent retrieval
  core. `personal_knowledge/research/` restores a bounded SpecRAG-derived
  Planner/Researcher/Merger/Synthesizer workflow behind a small chat-model
  abstraction. Neither package knows about LangChain Tools, Deep Agents, CLI,
  authentication, MCP, or HTTP services.

## What `memory` means in the reference project

The two reference `memory` directories have different and narrower meanings:

- `backend/memory/` contains `TypedDict` workflow state (`ResearchState` and
  `DraftState`). It is transient graph state, not durable user memory.
- `gpt_researcher/memory/` contains the embedding-provider abstraction used for
  similarity search. It is not conversation history or a user profile.
- Context filtering and compression are under `gpt_researcher/context/`, not
  under either `memory/` directory.
- Report/chat persistence is handled separately by backend storage code.

This project therefore does not create a `memory/` directory yet. The embedded
private corpus and its indexes are knowledge retrieval, not Agent memory. If a
future requirement introduces durable conversation or user memory, it should
be designed as its own backend capability with an explicit retention and
privacy model.

## Rules for future folders

1. Add `frontend/` only with an independently runnable frontend application.
2. Add `evals/` when repeatable evaluation datasets and runners exist.
3. Add `mcp-server/` only if MCP becomes a supported external interface; the
   Personal Knowledge integration remains an ordinary local Python Tool.
4. Do not add workspace, login, Docker, or Terraform boundaries until the
   product actually needs those operational capabilities.
5. Keep the local `gpt-researcher-main/` checkout as a read-only reference; the
   installed, pinned package remains the runtime dependency.

## Path boundaries

The runtime has three deliberately separate path namespaces:

- Project paths: `.env` and relative `OUTPUTS_DIR` values are anchored to the
  repository root, not the shell's current working directory.
- Agent paths: Deep Agents sees one run directory as virtual `/`. It may write
  only `/sections/...` and `/report.md`; host drive letters, `outputs/`, parent
  traversal, and paths outside that run are rejected by `FilesystemBackend`.
- Personal Knowledge paths: `PERSONAL_KNOWLEDGE_ROOT` may point outside this
  repository (currently `F:\specrag`). Corpus, index, and evidence source paths
  must resolve beneath that root. These host paths are never exposed as Agent
  filesystem paths.

Every run writes lifecycle state to `run.json`. Once the Agent returns, a
bounded `agent-trace.json` records message roles, tool calls, truncated tool
results, and the final summary for local diagnosis; it is under ignored
`outputs/` and must not contain provider credentials.
