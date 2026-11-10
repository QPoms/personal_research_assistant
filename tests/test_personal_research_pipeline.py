from types import SimpleNamespace
import pytest

from personal_knowledge.contracts import NativeFunctionCall
from personal_knowledge.research.pipeline import (
    LocalResearchPipeline,
    LocalRetrievalToolRegistry,
    LocalTaskResearcher,
)
from personal_knowledge.retrieval.models import CorpusScope
from personal_knowledge.tools import (
    RetrievalToolRequest,
    RetrievalToolResult,
    ToolEvidence,
    parse_retrieval_tool_arguments,
    retrieval_tool_definitions,
)


def _evidence(evidence_id: str, excerpt: str) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "source_type": "personal",
        "title": "3GPP TS 24.501",
        "specification": "24.501",
        "release": 17,
        "version": "17.10.0",
        "locator": "Release 17 / 5.4 NAS security / Lines 10-12",
        "heading_path": ["5.4", "NAS security"],
        "line_start": 10,
        "line_end": 12,
        "excerpt": excerpt,
        "source_path": "data/3gpp/24.501.md",
        "document_sha256": "a" * 64,
        "semantic_rank": 1,
        "lexical_rank": 1,
        "rrf_score": 0.03,
        "source_valid": True,
    }


class FakeCore:
    cfg = SimpleNamespace(target_release=17)
    max_results = 6

    def __init__(self):
        self.calls = []
        self.registry = FakeRegistry(self)
        scope = CorpusScope(release=17)
        self.retriever = SimpleNamespace(
            corpus_scope=scope,
            documents=(SimpleNamespace(specification="24.501"),),
        )

    def build_tool_registry(self):
        return self.registry

    def _get_retriever(self):
        return self.retriever

    @staticmethod
    def validate_tool_evidence(item: ToolEvidence):
        evidence_id = (
            "E-EXCEPTION" if item.chunk_id == "chunk-exception" else "E-SHARED"
        )
        value = _evidence(evidence_id, item.excerpt)
        value.update(
            {
                "chunk_id": item.chunk_id,
                "retrieval_tool": item.match_kind,
                "score": item.score,
                "score_kind": item.score_kind,
            }
        )
        return value


class FakeRegistry:
    definitions = retrieval_tool_definitions()

    def __init__(self, core):
        self.core = core

    def parse_call(self, call: NativeFunctionCall):
        if call.name not in {item.name for item in self.definitions}:
            raise KeyError(call.name)
        return parse_retrieval_tool_arguments(call.name, call.arguments)

    def invoke(self, name: str, request: RetrievalToolRequest):
        self.core.calls.append((name, request.query))
        exception = "exception" in request.query
        items = [self._tool_evidence(request, "chunk-shared", 1)]
        if exception:
            items.append(self._tool_evidence(request, "chunk-exception", 2))
        return RetrievalToolResult(
            request_id=request.request_id,
            query_id=request.query_id,
            tool_name=name,
            status="completed",
            corpus_scope=self.core.retriever.corpus_scope,
            evidence=tuple(items),
        )

    @staticmethod
    def _tool_evidence(request, chunk_id, rank):
        exception = chunk_id == "chunk-exception"
        return ToolEvidence(
            evidence_id=f"{request.request_id}:{rank}",
            chunk_id=chunk_id,
            rank=rank,
            match_kind="semantic_search" if exception else "keyword_search",
            score=1.0 / rank,
            score_kind="bm25",
            specification="24.501",
            release=17,
            version="17.10.0",
            source_path="data/3gpp/24.501.md",
            document_sha256="A" * 64,
            heading_path=("5.4", "NAS security"),
            line_start=10,
            line_end=12,
            excerpt=(
                "A narrowly defined exception applies before security activation."
                if exception
                else "Integrity protection applies to protected NAS messages."
            ),
        )


class FakeStructuredModel:
    def complete_json(self, prompt: str):
        if "local private-knowledge research planner" in prompt:
            return {
                "tasks": [
                    {
                        "task_id": "T1",
                        "objective": "Establish the primary rule",
                        "question": "Find the NAS integrity protection rule",
                        "dependencies": [],
                    },
                    {
                        "task_id": "T2",
                        "objective": "Establish exceptions",
                        "question": "Find the NAS integrity protection exception",
                        "dependencies": ["T1"],
                    },
                ]
            }
        if (
            "SpecRAG local Researcher" in prompt
            and '"task_id":"T1"' in prompt
            and "Evidence ledger: []" in prompt
        ):
            return {
                "actions": [
                    {
                        "tool": "keyword_search",
                        "arguments": {
                            "query_id": "T1",
                            "max_results": 6,
                            "target_specifications": [],
                            "lexical_query": "NAS integrity rule",
                        },
                    }
                ],
                "active_evidence_ids": [],
                "submit_research_result": False,
            }
        if (
            "SpecRAG local Researcher" in prompt
            and '"task_id":"T2"' in prompt
            and "Evidence ledger: []" in prompt
        ):
            return {
                "actions": [
                    {
                        "tool": "semantic_search",
                        "arguments": {
                            "query_id": "T2",
                            "max_results": 6,
                            "target_specifications": [],
                            "semantic_query": "NAS integrity exception",
                        },
                    }
                ],
                "active_evidence_ids": [],
                "submit_research_result": False,
            }
        if "SpecRAG local Researcher" in prompt and '"task_id":"T1"' in prompt:
            return {
                "status": "sufficient",
                "active_evidence_ids": ["E-SHARED"],
                "unresolved_gaps": [],
                "submit_research_result": True,
                "actions": [],
            }
        if "SpecRAG local Researcher" in prompt and '"task_id":"T2"' in prompt:
            return {
                "status": "sufficient",
                "active_evidence_ids": ["E-SHARED", "E-EXCEPTION"],
                "unresolved_gaps": [],
                "submit_research_result": True,
                "actions": [],
            }
        if "grounded local-evidence synthesizer" in prompt:
            return {
                "findings": [
                    {
                        "task_ids": ["T1"],
                        "claim": "Protected NAS messages require integrity protection.",
                        "evidence_ids": ["E-SHARED"],
                        "status": "supported",
                    },
                    {
                        "task_ids": ["T2"],
                        "claim": "The corpus defines a pre-activation exception.",
                        "evidence_ids": ["E-EXCEPTION"],
                        "status": "supported",
                    },
                ],
                "conflicts": [],
                "unresolved_gaps": [],
                "local_summary": "The rule and its bounded exception are both supported.",
            }
        raise AssertionError("unexpected prompt")


def test_pipeline_preserves_planner_tasks_and_merges_selected_evidence():
    core = FakeCore()
    pipeline = LocalResearchPipeline(
        core,
        FakeStructuredModel(),
        max_tasks=3,
        max_rounds=2,
        max_parallel_tasks=2,
    )

    result = pipeline.research(
        "When does NAS integrity protection apply, and what are the exceptions?"
    )

    assert result.status == "completed"
    assert [task.task_id for task in result.plan.tasks] == ["T1", "T2"]
    assert result.plan.tasks[1].dependencies == ("T1",)
    assert [item.task_id for item in result.task_results] == ["T1", "T2"]
    assert len(result.evidence) == 2
    shared = next(item for item in result.evidence if item["evidence_id"] == "E-SHARED")
    assert shared["task_ids"] == ["T1", "T2"]
    assert [finding.finding_id for finding in result.findings] == ["F1", "F2"]
    assert result.local_summary.startswith("The rule")
    assert core.calls == [
        ("keyword_search", "NAS integrity rule"),
        ("semantic_search", "NAS integrity exception"),
    ]
    assert result.task_results[0].tool_calls[0]["tool"] == "keyword_search"
    assert result.task_results[1].tool_calls[0]["tool"] == "semantic_search"
    assert len(pipeline.tools.definitions) == 9
    assert result.task_results[1].active_evidence_ids == ("E-SHARED", "E-EXCEPTION")


def test_registry_rejects_model_invented_query_id():
    tools = LocalRetrievalToolRegistry(FakeCore())

    with pytest.raises(ValueError, match="unknown query_id: q1; expected T1"):
        tools.parse_action(
            {
                "tool": "keyword_search",
                "arguments": {
                    "query_id": "q1",
                    "max_results": 6,
                    "target_specifications": [],
                    "lexical_query": "NAS integrity",
                },
            },
            call_id="T1-R1-C1",
            query_id="T1",
        )


def test_call_signature_ignores_only_result_limit():
    first = parse_retrieval_tool_arguments(
        "keyword_search",
        {
            "query_id": "T1",
            "max_results": 4,
            "target_specifications": ["24.501"],
            "lexical_query": "NAS integrity",
        },
    )
    second = parse_retrieval_tool_arguments(
        "keyword_search",
        {
            "query_id": "T1",
            "max_results": 12,
            "target_specifications": ["24.501"],
            "lexical_query": " nas INTEGRITY ",
        },
    )

    assert LocalTaskResearcher._call_signature(
        "keyword_search", first
    ) == LocalTaskResearcher._call_signature("keyword_search", second)


class DependencyNavigationModel:
    def complete_json(self, prompt: str):
        if "Evidence ledger: []" in prompt:
            return {
                "active_evidence_ids": [],
                "submit_research_result": False,
                "actions": [
                    {
                        "tool": "expand_source_context",
                        "arguments": {
                            "query_id": "T2",
                            "max_results": 6,
                            "seed_evidence_ids": ["E-DEPENDENCY"],
                        },
                    }
                ],
            }
        return {
            "status": "sufficient",
            "active_evidence_ids": ["E-SHARED"],
            "evidence_requirements": [],
            "unresolved_gaps": [],
            "submit_research_result": True,
            "actions": [],
        }


def test_dependency_evidence_is_a_trusted_navigation_seed_not_a_candidate():
    core = FakeCore()
    researcher = LocalTaskResearcher(
        DependencyNavigationModel(),
        LocalRetrievalToolRegistry(core),
        max_rounds=2,
        max_tool_calls=2,
    )
    dependency = _evidence("E-DEPENDENCY", "Dependency evidence.")
    dependency.update(
        {
            "chunk_id": "chunk-shared",
            "retrieval_tool": "keyword_search",
            "score": 1.0,
            "score_kind": "bm25",
        }
    )

    product = researcher.research(
        SimpleNamespace(
            task_id="T2",
            objective="Follow the dependency evidence",
            question="Find adjacent context in TS 24.501",
            model_dump_json=lambda: '{"task_id":"T2"}',
        ),
        dependency_context=[dependency],
    )

    assert product.result.status == "sufficient"
    assert product.result.candidate_count == 1
    assert product.result.tool_calls[0]["tool"] == "expand_source_context"
    assert product.result.tool_calls[0]["status"] == "completed"
