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
from typing import Any, Callable

from personal_knowledge.service import LocalPersonalKnowledgeService

from .model import StructuredResearchModel
from .models import (
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
as one task. Produce between 1 and {self.max_tasks} ordered tasks. Dependencies
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


@dataclass(slots=True)
class _ResearcherWorkProduct:
    result: TaskResearchResult
    selected_evidence: tuple[dict[str, Any], ...]


class LocalTaskResearcher:
    def __init__(
        self,
        model: StructuredResearchModel,
        search: Callable[[str], dict[str, Any]],
        *,
        max_rounds: int = 2,
        max_selected_evidence: int = 6,
    ) -> None:
        self.model = model
        self.search = search
        self.max_rounds = max_rounds
        self.max_selected_evidence = max_selected_evidence

    @staticmethod
    def _merge_candidates(
        existing: dict[str, dict[str, Any]],
        payload: dict[str, Any],
    ) -> tuple[str, ...]:
        errors = tuple(str(item) for item in payload.get("unresolved_gaps") or ())
        for item in payload.get("evidence") or ():
            if not isinstance(item, dict):
                continue
            evidence_id = item.get("evidence_id")
            if isinstance(evidence_id, str) and item.get("source_valid") is True:
                existing.setdefault(evidence_id, dict(item))
        return errors

    def _assess(
        self,
        task: ResearchTask,
        candidates: list[dict[str, Any]],
        *,
        dependency_context: list[dict[str, Any]],
        allow_followup: bool,
    ) -> dict[str, Any]:
        prompt = f"""You are the local knowledge Researcher for exactly one task.
Select only evidence that directly supports the task. Decide whether the
selected evidence is sufficient. If important coverage is missing and a
follow-up is allowed, propose one focused retrieval query. Do not invent facts,
citations, specification numbers, or evidence IDs.

Return JSON only:
{{
  "status":"sufficient|partial|insufficient",
  "supporting_evidence_ids":["E-..."],
  "unresolved_gaps":["..."],
  "followup_query":"... or null"
}}

Task: {task.model_dump_json()}
Dependency context: {json.dumps(_compact_evidence(dependency_context, limit=6), ensure_ascii=False)}
Candidate evidence: {json.dumps(_compact_evidence(candidates), ensure_ascii=False)}
Follow-up allowed: {str(allow_followup).lower()}
"""
        return self.model.complete_json(prompt)

    def research(
        self,
        task: ResearchTask,
        *,
        dependency_context: list[dict[str, Any]],
    ) -> _ResearcherWorkProduct:
        candidates: dict[str, dict[str, Any]] = {}
        gaps: list[str] = []
        rounds = 1
        first = self.search(task.question)
        gaps.extend(self._merge_candidates(candidates, first))

        if not candidates:
            result = TaskResearchResult(
                task_id=task.task_id,
                objective=task.objective,
                status="insufficient",
                supporting_evidence_ids=(),
                unresolved_gaps=tuple(dict.fromkeys(gaps or ["No source-verified evidence was retrieved."])),
                retrieval_rounds=rounds,
                candidate_count=0,
            )
            return _ResearcherWorkProduct(result=result, selected_evidence=())

        assessment_error: str | None = None
        try:
            assessment = self._assess(
                task,
                list(candidates.values()),
                dependency_context=dependency_context,
                allow_followup=self.max_rounds > 1,
            )
        except Exception as exc:
            assessment_error = f"researcher_fallback: {type(exc).__name__}: {exc}"
            assessment = {}

        followup = assessment.get("followup_query")
        if (
            isinstance(followup, str)
            and followup.strip()
            and rounds < self.max_rounds
            and assessment.get("status") != "sufficient"
        ):
            rounds += 1
            second = self.search(followup.strip())
            gaps.extend(self._merge_candidates(candidates, second))
            try:
                assessment = self._assess(
                    task,
                    list(candidates.values()),
                    dependency_context=dependency_context,
                    allow_followup=False,
                )
                assessment_error = None
            except Exception as exc:
                assessment_error = f"researcher_fallback: {type(exc).__name__}: {exc}"
                assessment = {}

        allowed_ids = set(candidates)
        raw_ids = assessment.get("supporting_evidence_ids") or ()
        selected_ids = tuple(dict.fromkeys(
            item for item in raw_ids if isinstance(item, str) and item in allowed_ids
        ))[: self.max_selected_evidence]
        raw_status = assessment.get("status")
        status = raw_status if raw_status in {"sufficient", "partial", "insufficient"} else "partial"
        assessment_gaps = assessment.get("unresolved_gaps") or ()
        gaps.extend(str(item) for item in assessment_gaps if str(item).strip())
        if assessment_error:
            gaps.append(assessment_error)

        if not selected_ids:
            selected_ids = tuple(candidates)[: self.max_selected_evidence]
            status = "partial"
            gaps.append("Researcher selection was unavailable; retained the highest-ranked verified evidence.")
        selected = tuple(candidates[evidence_id] for evidence_id in selected_ids)
        result = TaskResearchResult(
            task_id=task.task_id,
            objective=task.objective,
            status=status,
            supporting_evidence_ids=selected_ids,
            unresolved_gaps=tuple(dict.fromkeys(gaps)),
            retrieval_rounds=rounds,
            candidate_count=len(candidates),
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
    ) -> None:
        self.core = core
        self.planner = LocalResearchPlanner(model, max_tasks=max_tasks)
        self.researcher = LocalTaskResearcher(
            model,
            core.search,
            max_rounds=max_rounds,
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
    "LocalTaskResearcher",
]
