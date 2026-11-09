from types import SimpleNamespace

from personal_knowledge.research.pipeline import LocalResearchPipeline


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

    def search(self, query: str):
        if "exception" in query:
            evidence = [
                _evidence("E-SHARED", "Integrity protection applies to protected NAS messages."),
                _evidence("E-EXCEPTION", "A narrowly defined exception applies before security activation."),
            ]
        else:
            evidence = [
                _evidence("E-SHARED", "Integrity protection applies to protected NAS messages."),
            ]
        return {
            "status": "completed",
            "evidence": evidence,
            "unresolved_gaps": [],
        }


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
        if "local knowledge Researcher" in prompt and '"task_id":"T1"' in prompt:
            return {
                "status": "sufficient",
                "supporting_evidence_ids": ["E-SHARED"],
                "unresolved_gaps": [],
                "followup_query": None,
            }
        if "local knowledge Researcher" in prompt and '"task_id":"T2"' in prompt:
            return {
                "status": "sufficient",
                "supporting_evidence_ids": ["E-SHARED", "E-EXCEPTION"],
                "unresolved_gaps": [],
                "followup_query": None,
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
    pipeline = LocalResearchPipeline(
        FakeCore(),
        FakeStructuredModel(),
        max_tasks=3,
        max_rounds=2,
        max_parallel_tasks=2,
    )

    result = pipeline.research("When does NAS integrity protection apply, and what are the exceptions?")

    assert result.status == "completed"
    assert [task.task_id for task in result.plan.tasks] == ["T1", "T2"]
    assert result.plan.tasks[1].dependencies == ("T1",)
    assert [item.task_id for item in result.task_results] == ["T1", "T2"]
    assert len(result.evidence) == 2
    shared = next(item for item in result.evidence if item["evidence_id"] == "E-SHARED")
    assert shared["task_ids"] == ["T1", "T2"]
    assert [finding.finding_id for finding in result.findings] == ["F1", "F2"]
    assert result.local_summary.startswith("The rule")
