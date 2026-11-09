# Personal Knowledge workflow

The embedded Personal Knowledge integration is an ordinary local Python Tool.
It is not MCP and does not call a SpecRAG HTTP service.

```text
Chief Editor
  -> delegates a private-knowledge section to a Researcher
    -> search_personal_knowledge(query)
      -> Local Planner (1-3 tasks, ordered dependencies)
      -> Local task Researchers
          -> BM25 + optional Dense retrieval
          -> RRF
          -> deterministic source/hash/locator/excerpt validation
          -> one optional focused follow-up retrieval
          -> selected supporting Evidence IDs
      -> deterministic cross-task Evidence Merger
      -> Local Evidence Synthesizer
          -> findings bound to Evidence IDs
          -> conflicts and unresolved gaps
          -> concise local summary
    <- LocalResearchResult
  -> Chief combines the local brief with Web research and writes the final report
```

## Removed boundary

The original separate model-based Sufficiency Judge/Validator loop is not part
of this runtime. Each task Researcher decides whether its evidence is
sufficient and may perform at most one follow-up retrieval. The inexpensive,
deterministic source-integrity validator remains mandatory.

The local Synthesizer is retained with a narrower contract: it prepares a
grounded evidence brief for the Chief Editor and cannot publish the final user
answer. It receives only source-verified, Researcher-selected evidence.

## Tool result

The Tool returns:

- the local research plan and dependencies;
- one result per task, including status and selected Evidence IDs;
- deduplicated selected evidence with every contributing task ID;
- grounded findings linked to Evidence IDs;
- conflicts, unresolved gaps, and pipeline warnings;
- a concise `local_summary` for the Chief Editor.

Unselected candidate chunks remain internal to the local workflow and are not
placed in the Chief Editor's context.

## Privacy gate

Planner, task Researcher, and local Synthesizer prompts contain selected local
evidence excerpts. When the configured model is a third-party hosted model,
this is an external disclosure of private corpus content. The application fails
closed unless:

```env
PERSONAL_KNOWLEDGE_ALLOW_EXTERNAL_MODEL=true
```

Keep this value `false` for unapproved providers. A future local model can use
the same structured-model boundary without changing the Tool contract.
