# Third-Party Notices

## GPT Researcher

- Project: GPT Researcher
- Upstream: <https://github.com/assafelovic/gpt-researcher>
- Package version: `0.16.0`, installed from the audited commit below because
  the PyPI 0.16.0 artifact has a broken import order in
  `actions/query_processing.py` (`Any` is evaluated before it is imported).
- Audited upstream commit: `bbb974b9b21a497c170efdf5346d55a1ac8e6d02`
- Local reference status: the supplied snapshot has no `.git` metadata; its
  representative core files match that upstream commit, but the entire
  snapshot cannot be cryptographically attributed to the commit.
- Copyright: GPT Researcher contributors, including Assaf Elovic

The Chief Editor/Researcher workflow and GPT Researcher tool adapter in this
repository are adapted from:

- `deep_agents/agent.py`
- `deep_agents/tools.py`

They were modified to add program-enforced Web/Personal/Mixed source modes,
SpecRAG tool access, bounded concurrency, safe configuration, and run-scoped
cost tracking. GPT Researcher's research engine itself is consumed as the
published package and is not copied into this project.

### License metadata discrepancy

At the audited source state, GPT Researcher's root `LICENSE` contains the
Apache License 2.0, while `pyproject.toml` and `setup.py` declare MIT. No root
`NOTICE` file was found. This project conservatively treats copied/adapted
source as Apache-2.0 and includes that license at
`LICENSES/Apache-2.0.txt`. The discrepancy requires human/legal review before
external distribution; this notice does not resolve or reinterpret it.

## Deep Agents

- Project: Deep Agents
- Package version used: `0.7.8`
- Upstream tag commit: `1e261ba201bb1af4dbc5cbc8b6424e709b850ea8`
- Used through its public package API; no engine source is vendored here.

## DDGS

- Package version used: `9.16.1`
- Purpose: explicit runtime dependency for GPT Researcher's DuckDuckGo
  Retriever. The audited GPT Researcher commit imports `ddgs` but does not
  declare it as a required dependency.

## SpecRAG

- Independent local service version: `0.1.0`
- Audited local commit: `8c161ea3a3c8b3139af21500192982536a269362`
- Integration: Bearer-authenticated HTTP API only
- No SpecRAG retrieval, LangGraph, or model implementation is copied here.

