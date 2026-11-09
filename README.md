# Personal Research Assistant

Personal Research Assistant coordinates two independent research sources at the
agent layer:

- **GPT Researcher 0.16.0** performs Web research through its public Python API.
- **SpecRAG** remains an independent HTTP service and is exposed as an ordinary
  LangChain tool for private, evidence-backed knowledge.
- **Deep Agents 0.7.8** supplies the Chief Editor harness, Researcher subagent,
  Todo management, filesystem tools, and task delegation.

The project does not reimplement GPT Researcher's engine, does not use its
local/hybrid retriever, and does not turn SpecRAG into MCP.

## Current architecture

```text
User query
    -> Chief Editor (scope, plan, delegate, review, assemble)
        -> Researcher subagents
            -> GPT Researcher [web only]
            -> SpecRAG HTTP Tool [personal only]
        -> outputs/run_<id>/sections/*.md
        -> outputs/run_<id>/report.md
```

The importable package lives directly at `research_assistant/`, mirroring the
upstream repository's root-level `gpt_researcher/` package. There is no extra
`src/` nesting.

Source authorization is enforced by code when the agents are constructed:

| Mode | Chief tools | Researcher tools |
|---|---|---|
| `web` | `quick_search` | `quick_search`, `deep_research` |
| `personal` | `search_personal_knowledge` | `search_personal_knowledge` |
| `mixed` | both scoping tools | all three tools |

## Requirements

- Python 3.12+
- `uv`
- Model and search-provider credentials supported by GPT Researcher 0.16.0
- For `personal` or `mixed`: a running SpecRAG service, workspace ID, and token

The repository pins top-level integrations in `pyproject.toml` and the complete
resolved graph in `uv.lock`. GPT Researcher is pinned to the audited upstream
commit because the PyPI 0.16.0 artifact currently fails during import.

## Setup

```powershell
uv sync --extra test
Copy-Item .env.example .env
```

Configure `.env`. At minimum, set the Deep Agents/model-provider values and the
search provider required by GPT Researcher. `RETRIEVER=duckduckgo` works
without a separate search key for development; Tavily requires its own key and
is generally the more production-oriented choice. For private knowledge, also set:

```env
SPECRAG_BASE_URL=http://127.0.0.1:8000
SPECRAG_WORKSPACE_ID=your-authorized-workspace
SPECRAG_ACCESS_TOKEN=your-token
```

The SpecRAG token and workspace are application configuration. They are never
tool arguments and are never placed in agent prompts.

## Run

```powershell
uv run research-assistant `
  --query "分析近年来 RAG 技术的发展现状" `
  --source-mode web
```

During repository development, the equivalent root-level entry point is
`uv run python cli.py ...`.

Available modes are `web`, `personal`, and `mixed`. Each run gets an isolated
directory under `outputs/` with `run.json`, section drafts, and `report.md`.

## Tests

```powershell
uv run pytest
```

The automated suite covers source authorization, evidence normalization, Web
engine contracts, SpecRAG asynchronous states, polling timeout, errors, prompt
variants, and run-path isolation. Live Web/SpecRAG acceptance tests require
real credentials and services and are intentionally separate from unit tests.

## Upstream reference

`gpt-researcher-main/` is a local-only source reference and is excluded by
`.gitignore`. It is not a runtime dependency and should be removed after the
implementation/audit work is complete. Provenance and the upstream license
metadata conflict are recorded in `THIRD_PARTY_NOTICES.md`.

