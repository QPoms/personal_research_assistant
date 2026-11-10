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
  research/                 # run orchestration and report validation
  runtime/                  # model construction
  storage/                  # isolated run directories
deep_agents/                # Chief Editor and Researcher team construction
personal_knowledge/         # embedded private-knowledge retrieval engine
  infrastructure/           # optional local vector-store adapter
  retrieval/                # corpus, BM25, dense, RRF, clauses, snapshots
  research/                 # Planner, task Researchers, merger, local brief
research_tools/             # Agent-facing Web/personal tools and policies
docs/                       # architecture decisions
tests/                      # unit and contract tests
cli.py                      # repository-level development entry point
```

There is currently no `frontend/`, `memory/`, `mcp-server/`, or `terraform/`
directory because the project does not yet implement those capabilities. New
top-level boundaries should be added only when executable code actually exists.
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
`agent-trace.json`, section drafts, and `report.md`. Relative output paths are
anchored to the repository root. Inside Deep Agents, the same run directory is
exposed only as virtual `/`; drafts use `/sections/...` and the final report
uses `/report.md`.

## Tests

```powershell
uv run pytest
```

The suite covers source authorization, evidence normalization, upstream Web
engine contracts, embedded retrieval and source verification, prompts, and
run-path isolation. Live Web acceptance requires provider credentials.

## Upstream reference

`gpt-researcher-main/` is a local-only source reference and is excluded by
`.gitignore`. It is not a runtime dependency and is retained locally for the
ongoing implementation audit. Provenance is recorded in
`THIRD_PARTY_NOTICES.md`.
