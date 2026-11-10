"""Bounded local research pipeline derived from SpecRAG's core workflow.

The retained workflow is Planner -> task Researchers -> deterministic evidence
merge -> grounded local synthesis. The former separate LLM sufficiency judge is
removed: each Researcher decides whether its own evidence is sufficient and may
issue at most one follow-up retrieval. Source integrity remains enforced by the
underlying retrieval service before evidence enters this pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import re
from typing import Any

from personal_knowledge.contracts import NativeFunctionCall
from personal_knowledge.service import LocalPersonalKnowledgeService
from personal_knowledge.tools import (
    ClauseLookupArguments,
    ClauseNavigationArguments,
    CrossReferenceLookupArguments,
    ExpandSourceContextArguments,
    HybridSearchArguments,
    HyDESearchArguments,
    KeywordSearchArguments,
    RerankArguments,
    RetrievalToolArguments,
    RetrievalToolRequest,
    SemanticSearchArguments,
    ToolRegistry,
)

from .model import StructuredResearchModel
from .models import (
    EvidenceRequirement,
    Finding,
    LocalResearchResult,
    ResearchPlan,
    ResearchTask,
    TaskResearchResult,
)


def _compact_evidence(items: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items[:limit]:
        compact.append({
            "evidence_id": item.get("evidence_id"),
            "title": item.get("title"),
            "specification": item.get("specification"),
            "release": item.get("release"),
            "version": item.get("version"),
            "locator": item.get("locator"),
            "excerpt": str(item.get("excerpt") or "")[:1200],
        })
    return compact


class LocalResearchPlanner:
    def __init__(self, model: StructuredResearchModel, *, max_tasks: int = 3) -> None:
        self.model = model
        self.max_tasks = max_tasks

    def plan(self, question: str) -> tuple[ResearchPlan, tuple[str, ...]]:
        prompt = f"""You are the local private-knowledge research planner.
Decompose the question only when it contains genuinely independent research
objectives. A cohesive question that one retrieval context can answer must stay
as one task. Explicitly enumerated normative dimensions such as applicability
conditions, exceptions, and failure handling should be separate tasks when they
are likely governed by different clauses or evidence. Produce between 1 and
{self.max_tasks} ordered tasks. Dependencies
may reference only earlier tasks and only when their validated evidence is
required to research the later task. Do not create a terminal writing,
integration, validation, or synthesis task.

Return JSON only:
{{"tasks":[{{"task_id":"T1","objective":"...","question":"...","dependencies":[]}}]}}

Question: {question}
"""
        try:
            payload = self.model.complete_json(prompt)
            tasks = payload.get("tasks")
            if not isinstance(tasks, list):
                raise ValueError("planner response has no task list")
            plan = ResearchPlan.model_validate({"tasks": tasks[: self.max_tasks]})
            return plan, ()
        except Exception as exc:
            fallback = ResearchPlan(tasks=(ResearchTask(
                task_id="T1",
                objective=question,
                question=question,
                dependencies=(),
            ),))
            return fallback, (f"planner_fallback: {type(exc).__name__}: {exc}",)


class LocalRetrievalToolRegistry:
    """Adapter over SpecRAG's original strict nine-tool registry."""

    def __init__(self, core: LocalPersonalKnowledgeService) -> None:
        self.core = core
        self.registry: ToolRegistry = core.build_tool_registry()
        self.definitions = tuple(
            item.model_dump(mode="json") for item in self.registry.definitions
        )
        retriever = core._get_retriever()
        self.available_specifications = frozenset(
            item.specification for item in retriever.documents
        )

    def parse_action(
        self,
        raw: Any,
        *,
        call_id: str,
        query_id: str,
    ) -> tuple[str, RetrievalToolArguments]:
        if not isinstance(raw, dict):
            raise ValueError("tool action must be an object")
        name = raw.get("tool")
        arguments = raw.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError("tool action requires tool and arguments")
        normalized = dict(arguments)
        normalized.setdefault("query_id", query_id)
        normalized.setdefault("max_results", min(self.core.max_results, 12))
        parsed = self.registry.parse_call(NativeFunctionCall(
            call_id=call_id,
            name=name,
            arguments=normalized,
        ))
        return name, parsed

    def execute(
        self,
        name: str,
        arguments: RetrievalToolArguments,
        *,
        request_id: str,
        candidates: dict[str, dict[str, Any]],
        grounded_specifications: frozenset[str],
    ) -> dict[str, Any]:
        targets = tuple(getattr(arguments, "target_specifications", ()))
        unknown = sorted(set(targets) - self.available_specifications)
        if unknown:
            raise ValueError("unknown specifications: " + ", ".join(unknown))
        ungrounded = sorted(set(targets) - grounded_specifications)
        if ungrounded:
            raise ValueError(
                "specifications are not grounded in the task, dependency "
                "context, or observed evidence: " + ", ".join(ungrounded)
            )

        seed_ids = tuple(getattr(arguments, "seed_evidence_ids", ()))
        unknown_seeds = tuple(item for item in seed_ids if item not in candidates)
        if unknown_seeds:
            raise ValueError("unknown seed evidence IDs: " + ", ".join(unknown_seeds))
        seed_chunks = tuple(str(candidates[item]["chunk_id"]) for item in seed_ids)

        query = ""
        semantic_query = None
        hypothetical_document = None
        clause = None
        relation = None
        reference_specification = None
        reference_version = None
        reference_clause = None
        include_descendants = False
        cursor = 0
        if isinstance(arguments, KeywordSearchArguments):
            query = arguments.lexical_query
        elif isinstance(arguments, SemanticSearchArguments):
            query = arguments.semantic_query
            semantic_query = arguments.semantic_query
        elif isinstance(arguments, HybridSearchArguments):
            query = arguments.lexical_query
            semantic_query = arguments.semantic_query
        elif isinstance(arguments, HyDESearchArguments):
            query = arguments.lexical_query
            hypothetical_document = arguments.hypothetical_document
        elif isinstance(arguments, RerankArguments):
            query = arguments.semantic_query
            semantic_query = arguments.semantic_query
        elif isinstance(arguments, ClauseLookupArguments):
            clause = arguments.clause
            include_descendants = arguments.include_descendants
            cursor = arguments.cursor
        elif isinstance(arguments, ClauseNavigationArguments):
            relation = arguments.relation
            cursor = arguments.cursor
        elif isinstance(arguments, CrossReferenceLookupArguments):
            reference_specification = arguments.target_specification
            reference_version = arguments.target_version
            reference_clause = arguments.target_clause
            cursor = arguments.cursor
        elif not isinstance(arguments, ExpandSourceContextArguments):
            raise TypeError(f"unsupported retrieval arguments: {type(arguments).__name__}")

        request = RetrievalToolRequest(
            request_id=request_id,
            query_id=arguments.query_id,
            query=query,
            semantic_query=semantic_query,
            hypothetical_document=hypothetical_document,
            target_specifications=targets,
            clause=clause,
            clause_relation=relation,
            reference_specification=reference_specification,
            reference_version=reference_version,
            reference_clause=reference_clause,
            include_descendants=include_descendants,
            cursor=cursor,
            seed_chunk_ids=seed_chunks,
            limit=arguments.max_results,
        )
        result = self.registry.invoke(name, request)  # type: ignore[arg-type]
        validated: list[dict[str, Any]] = []
        errors = list(result.errors)
        for item in result.evidence:
            evidence = self.core.validate_tool_evidence(item)
            if evidence["source_valid"]:
                validated.append(evidence)
            else:
                errors.append(
                    f"source validation rejected chunk {item.chunk_id}"
                )
        status = result.status
        if validated and errors:
            status = "partial"
        elif not validated and status == "completed":
            status = "unavailable"
        return {
            "status": status,
            "evidence": validated,
            "errors": errors,
            "next_cursor": result.next_cursor,
        }


@dataclass(slots=True)
class _ResearcherWorkProduct:
    result: TaskResearchResult
    selected_evidence: tuple[dict[str, Any], ...]


class LocalTaskResearcher:
    def __init__(
        self,
        model: StructuredResearchModel,
        tools: LocalRetrievalToolRegistry,
        *,
        max_rounds: int = 2,
        max_selected_evidence: int = 6,
        max_tool_calls: int = 4,
    ) -> None:
        self.model = model
        self.tools = tools
        self.max_rounds = max_rounds
        self.max_selected_evidence = max_selected_evidence
        self.max_tool_calls = max_tool_calls

    def _review(
        self,
        task: ResearchTask,
        candidates: dict[str, dict[str, Any]],
        *,
        active_ids: tuple[str, ...],
        dependency_context: list[dict[str, Any]],
        allow_actions: bool,
        remaining_tool_calls: int,
    ) -> dict[str, Any]:
        active = set(active_ids)
        evidence_payload = []
        for evidence_id, item in candidates.items():
            value = dict(_compact_evidence([item], limit=1)[0])
            value["status"] = "active" if evidence_id in active else "candidate"
            value["retrieval_tool"] = item.get("retrieval_tool")
            evidence_payload.append(value)
        prompt = f"""You are the SpecRAG local Researcher for exactly one task.
Maintain a bounded active evidence set while retaining all observed candidates.
Review evidence before acting. You may invoke the original retrieval tools,
including clause lookup/navigation, cross-reference traversal, source-context
expansion, HyDE, and reranking. Seed arguments always use observed Evidence IDs,
never chunk IDs. Target specifications must be grounded in the task, dependency
evidence, or observed evidence. Do not invent citations or clause numbers.

This Researcher performs its own sufficiency decision. There is no downstream
Validator. Set submit_research_result=true only after choosing the evidence that
should be sent to the deterministic merger and local synthesizer.

Return JSON only:
{{
  "assessment":"...",
  "status":"sufficient|partial|insufficient",
  "evidence_requirements":[{{"requirement_id":"R1","description":"...","coverage_status":"missing|partial|covered","evidence_ids":["E-..."]}}],
  "active_evidence_ids":["E-..."],
  "unresolved_gaps":["..."],
  "submit_research_result":false,
  "actions":[{{"tool":"one registered tool name","arguments":{{...}}}}]
}}

Registered tool definitions: {json.dumps(self.tools.definitions, ensure_ascii=False)}
Task: {task.model_dump_json()}
Dependency context: {json.dumps(_compact_evidence(dependency_context, limit=6), ensure_ascii=False)}
Evidence ledger: {json.dumps(evidence_payload, ensure_ascii=False)}
Actions allowed: {str(allow_actions).lower()}
Remaining tool-call budget: {remaining_tool_calls}
Maximum active evidence: {self.max_selected_evidence}
"""
        return self.model.complete_json(prompt)

    @staticmethod
    def _grounded_specifications(
        task: ResearchTask,
        dependency_context: list[dict[str, Any]],
        candidates: dict[str, dict[str, Any]],
    ) -> frozenset[str]:
        explicit = set(re.findall(r"\b\d{2}\.\d{3}\b", task.question))
        explicit.update(
            str(item["specification"])
            for item in (*dependency_context, *candidates.values())
            if item.get("specification")
        )
        return frozenset(explicit)

    def _fallback_action(self, task: ResearchTask) -> dict[str, Any]:
        return {
            "tool": "hybrid_search",
            "arguments": {
                "query_id": task.task_id,
                "max_results": min(self.tools.core.max_results, 12),
                "target_specifications": sorted(
                    set(re.findall(r"\b\d{2}\.\d{3}\b", task.question))
                ),
                "lexical_query": task.question,
                "semantic_query": task.question,
            },
        }

    @staticmethod
    def _select_ids(
        raw: Any,
        candidates: dict[str, dict[str, Any]],
        *,
        limit: int,
    ) -> tuple[str, ...]:
        if not isinstance(raw, list):
            return ()
        return tuple(dict.fromkeys(
            item for item in raw if isinstance(item, str) and item in candidates
        ))[:limit]

    @staticmethod
    def _requirements(
        raw: Any,
        candidates: dict[str, dict[str, Any]],
    ) -> tuple[EvidenceRequirement, ...]:
        if not isinstance(raw, list):
            return ()
        requirements: list[EvidenceRequirement] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized["evidence_ids"] = tuple(
                evidence_id
                for evidence_id in normalized.get("evidence_ids") or ()
                if evidence_id in candidates
            )
            if normalized.get("coverage_status") != "missing" and not normalized["evidence_ids"]:
                normalized["coverage_status"] = "missing"
            try:
                requirements.append(EvidenceRequirement.model_validate(normalized))
            except ValueError:
                continue
        return tuple(requirements)

    def research(
        self,
        task: ResearchTask,
        *,
        dependency_context: list[dict[str, Any]],
    ) -> _ResearcherWorkProduct:
        candidates: dict[str, dict[str, Any]] = {}
        gaps: list[str] = []
        rounds = 0
        tool_calls: list[dict[str, Any]] = []
        active_ids: tuple[str, ...] = ()
        final_assessment: dict[str, Any] = {}
        submitted = False

        while rounds < self.max_rounds and len(tool_calls) < self.max_tool_calls:
            remaining = self.max_tool_calls - len(tool_calls)
            try:
                review = self._review(
                    task,
                    candidates,
                    active_ids=active_ids,
                    dependency_context=dependency_context,
                    allow_actions=True,
                    remaining_tool_calls=remaining,
                )
            except Exception as exc:
                gaps.append(f"researcher_fallback: {type(exc).__name__}: {exc}")
                review = {"actions": [self._fallback_action(task)]} if not candidates else {}

            selected = self._select_ids(
                review.get("active_evidence_ids"),
                candidates,
                limit=self.max_selected_evidence,
            )
            if selected or review.get("active_evidence_ids") == []:
                active_ids = selected
            final_assessment = review
            if review.get("submit_research_result") is True and active_ids:
                submitted = True
                break

            raw_actions = review.get("actions")
            if not isinstance(raw_actions, list) or not raw_actions:
                if candidates:
                    break
                raw_actions = [self._fallback_action(task)]

            rounds += 1
            executed = 0
            for index, raw_action in enumerate(raw_actions[:remaining], 1):
                call_id = f"{task.task_id}-R{rounds}-C{index}"
                audit: dict[str, Any] = {"call_id": call_id}
                try:
                    name, arguments = self.tools.parse_action(
                        raw_action,
                        call_id=call_id,
                        query_id=task.task_id,
                    )
                    audit.update({
                        "tool": name,
                        "arguments": arguments.model_dump(mode="json"),
                    })
                    payload = self.tools.execute(
                        name,
                        arguments,
                        request_id=call_id,
                        candidates=candidates,
                        grounded_specifications=self._grounded_specifications(
                            task, dependency_context, candidates
                        ),
                    )
                    new_ids: list[str] = []
                    observed_ids: list[str] = []
                    for item in payload["evidence"]:
                        evidence_id = str(item["evidence_id"])
                        observed_ids.append(evidence_id)
                        if evidence_id not in candidates:
                            candidates[evidence_id] = item
                            new_ids.append(evidence_id)
                    gaps.extend(str(item) for item in payload["errors"])
                    audit.update({
                        "status": payload["status"],
                        "evidence_ids": observed_ids,
                        "new_evidence_ids": new_ids,
                        "next_cursor": payload["next_cursor"],
                        "errors": payload["errors"],
                    })
                except Exception as exc:
                    audit.update({
                        "tool": raw_action.get("tool") if isinstance(raw_action, dict) else None,
                        "status": "rejected",
                        "evidence_ids": [],
                        "new_evidence_ids": [],
                        "errors": [f"{type(exc).__name__}: {exc}"],
                    })
                    gaps.extend(audit["errors"])
                tool_calls.append(audit)
                executed += 1
            if executed == 0:
                break

        if candidates and not submitted:
            try:
                final_assessment = self._review(
                    task,
                    candidates,
                    active_ids=active_ids,
                    dependency_context=dependency_context,
                    allow_actions=False,
                    remaining_tool_calls=0,
                )
                selected = self._select_ids(
                    final_assessment.get("active_evidence_ids"),
                    candidates,
                    limit=self.max_selected_evidence,
                )
                if selected:
                    active_ids = selected
            except Exception as exc:
                gaps.append(f"researcher_finalization_fallback: {type(exc).__name__}: {exc}")

        if not candidates:
            result = TaskResearchResult(
                task_id=task.task_id,
                objective=task.objective,
                status="insufficient",
                supporting_evidence_ids=(),
                unresolved_gaps=tuple(dict.fromkeys(gaps or ["No source-verified evidence was retrieved."])),
                retrieval_rounds=max(rounds, 1),
                candidate_count=0,
                active_evidence_ids=(),
                evidence_requirements=(),
                tool_calls=tuple(tool_calls),
            )
            return _ResearcherWorkProduct(result=result, selected_evidence=())

        selected_ids = active_ids
        requirements = self._requirements(
            final_assessment.get("evidence_requirements"), candidates
        )
        raw_status = final_assessment.get("status")
        status = raw_status if raw_status in {"sufficient", "partial", "insufficient"} else "partial"
        assessment_gaps = final_assessment.get("unresolved_gaps") or ()
        gaps.extend(str(item) for item in assessment_gaps if str(item).strip())

        if not selected_ids:
            selected_ids = tuple(candidates)[: self.max_selected_evidence]
            status = "partial"
            gaps.append("Researcher active-set selection was unavailable; retained the highest-ranked verified evidence.")
        selected = tuple(candidates[evidence_id] for evidence_id in selected_ids)
        result = TaskResearchResult(
            task_id=task.task_id,
            objective=task.objective,
            status=status,
            supporting_evidence_ids=selected_ids,
            unresolved_gaps=tuple(dict.fromkeys(gaps)),
            retrieval_rounds=max(rounds, 1),
            candidate_count=len(candidates),
            active_evidence_ids=selected_ids,
            evidence_requirements=requirements,
            tool_calls=tuple(tool_calls),
        )
        return _ResearcherWorkProduct(result=result, selected_evidence=selected)


class EvidenceMerger:
    """Deterministically deduplicate selected evidence across research tasks."""

    def merge(self, products: list[_ResearcherWorkProduct]) -> tuple[dict[str, Any], ...]:
        merged: dict[str, dict[str, Any]] = {}
        for product in products:
            for item in product.selected_evidence:
                evidence_id = str(item["evidence_id"])
                if evidence_id not in merged:
                    merged[evidence_id] = dict(item)
                    merged[evidence_id]["task_ids"] = [product.result.task_id]
                elif product.result.task_id not in merged[evidence_id]["task_ids"]:
                    merged[evidence_id]["task_ids"].append(product.result.task_id)
        return tuple(merged.values())


class LocalEvidenceSynthesizer:
    """Produce a grounded local brief, never the user-facing final report."""

    def __init__(self, model: StructuredResearchModel) -> None:
        self.model = model

    def synthesize(
        self,
        question: str,
        plan: ResearchPlan,
        results: tuple[TaskResearchResult, ...],
        evidence: tuple[dict[str, Any], ...],
    ) -> tuple[tuple[Finding, ...], tuple[str, ...], tuple[str, ...], str, tuple[str, ...]]:
        eligible_ids = {str(item["evidence_id"]) for item in evidence}
        task_ids = {task.task_id for task in plan.tasks}
        prompt = f"""You are the grounded local-evidence synthesizer. Produce a
compact research brief for a Chief Editor. You are not writing the final user
answer. Every finding must cite only supplied evidence IDs. Preserve unresolved
gaps and conflicts. Never add unsupported facts.

Return JSON only:
{{
  "findings":[{{"task_ids":["T1"],"claim":"...","evidence_ids":["E-..."],"status":"supported|partial"}}],
  "conflicts":["..."],
  "unresolved_gaps":["..."],
  "local_summary":"..."
}}

Question: {question}
Plan: {plan.model_dump_json()}
Task results: {json.dumps([item.model_dump() for item in results], ensure_ascii=False)}
Selected evidence: {json.dumps(_compact_evidence(list(evidence), limit=18), ensure_ascii=False)}
"""
        warnings: list[str] = []
        try:
            payload = self.model.complete_json(prompt)
        except Exception as exc:
            warnings.append(f"synthesizer_fallback: {type(exc).__name__}: {exc}")
            payload = {}

        findings: list[Finding] = []
        for raw in payload.get("findings") or ():
            if not isinstance(raw, dict):
                continue
            finding_tasks = tuple(dict.fromkeys(
                item for item in raw.get("task_ids") or () if item in task_ids
            ))
            finding_evidence = tuple(dict.fromkeys(
                item for item in raw.get("evidence_ids") or () if item in eligible_ids
            ))
            claim = raw.get("claim")
            if not finding_tasks or not finding_evidence or not isinstance(claim, str) or not claim.strip():
                continue
            status = raw.get("status") if raw.get("status") in {"supported", "partial"} else "supported"
            findings.append(Finding(
                finding_id=f"F{len(findings) + 1}",
                task_ids=finding_tasks,
                claim=claim.strip(),
                evidence_ids=finding_evidence,
                status=status,
            ))

        if not findings:
            for result in results:
                task_evidence = [
                    item for item in evidence if result.task_id in item.get("task_ids", ())
                ]
                if not task_evidence:
                    continue
                item = task_evidence[0]
                excerpt = " ".join(str(item.get("excerpt") or "").split())[:700]
                findings.append(Finding(
                    finding_id=f"F{len(findings) + 1}",
                    task_ids=(result.task_id,),
                    claim=excerpt,
                    evidence_ids=(str(item["evidence_id"]),),
                    status="supported" if result.status == "sufficient" else "partial",
                ))
            warnings.append("Synthesizer returned no valid grounded findings; used an evidence digest.")

        conflicts = tuple(str(item) for item in payload.get("conflicts") or () if str(item).strip())
        model_gaps = tuple(str(item) for item in payload.get("unresolved_gaps") or () if str(item).strip())
        task_gaps = tuple(gap for result in results for gap in result.unresolved_gaps)
        gaps = tuple(dict.fromkeys((*task_gaps, *model_gaps)))
        summary = payload.get("local_summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = " ".join(finding.claim for finding in findings)[:2000]
        return tuple(findings), conflicts, gaps, summary.strip(), tuple(warnings)


class LocalResearchPipeline:
    def __init__(
        self,
        core: LocalPersonalKnowledgeService,
        model: StructuredResearchModel,
        *,
        max_tasks: int = 3,
        max_rounds: int = 2,
        max_parallel_tasks: int = 3,
        max_tool_calls_per_task: int = 4,
    ) -> None:
        self.core = core
        self.planner = LocalResearchPlanner(model, max_tasks=max_tasks)
        self.tools = LocalRetrievalToolRegistry(core)
        self.researcher = LocalTaskResearcher(
            model,
            self.tools,
            max_rounds=max_rounds,
            max_tool_calls=max_tool_calls_per_task,
        )
        self.merger = EvidenceMerger()
        self.synthesizer = LocalEvidenceSynthesizer(model)
        self.max_parallel_tasks = max_parallel_tasks

    def _dependency_context(
        self,
        task: ResearchTask,
        completed: dict[str, _ResearcherWorkProduct],
    ) -> list[dict[str, Any]]:
        return [
            item
            for dependency in task.dependencies
            for item in completed[dependency].selected_evidence
        ]

    def research(self, question: str) -> LocalResearchResult:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        plan, planner_warnings = self.planner.plan(question)
        completed: dict[str, _ResearcherWorkProduct] = {}
        pending = list(plan.tasks)

        while pending:
            ready = [
                task for task in pending
                if all(dependency in completed for dependency in task.dependencies)
            ]
            if not ready:
                raise RuntimeError("local research plan contains unresolved dependencies")
            with ThreadPoolExecutor(max_workers=min(self.max_parallel_tasks, len(ready))) as pool:
                futures = {
                    task.task_id: pool.submit(
                        self.researcher.research,
                        task,
                        dependency_context=self._dependency_context(task, completed),
                    )
                    for task in ready
                }
                for task in ready:
                    completed[task.task_id] = futures[task.task_id].result()
            ready_ids = {task.task_id for task in ready}
            pending = [task for task in pending if task.task_id not in ready_ids]

        ordered_products = [completed[task.task_id] for task in plan.tasks]
        results = tuple(product.result for product in ordered_products)
        evidence = self.merger.merge(ordered_products)
        findings, conflicts, gaps, summary, synthesis_warnings = self.synthesizer.synthesize(
            question,
            plan,
            results,
            evidence,
        )
        complete = bool(results) and all(result.status == "sufficient" for result in results)
        status = "completed" if complete and evidence else "partial" if evidence else "unavailable"
        return LocalResearchResult(
            status=status,
            query=question,
            release=self.core.cfg.target_release,
            plan=plan,
            task_results=results,
            findings=findings,
            evidence=evidence,
            conflicts=conflicts,
            unresolved_gaps=gaps,
            local_summary=summary,
            pipeline_warnings=(*planner_warnings, *synthesis_warnings),
        )


__all__ = [
    "EvidenceMerger",
    "LocalEvidenceSynthesizer",
    "LocalResearchPipeline",
    "LocalResearchPlanner",
    "LocalRetrievalToolRegistry",
    "LocalTaskResearcher",
]
