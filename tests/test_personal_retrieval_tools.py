from __future__ import annotations

from types import SimpleNamespace

from personal_knowledge.contracts import NativeFunctionCall
from personal_knowledge.retrieval.clauses import ClauseIndex
from personal_knowledge.retrieval.models import (
    ClauseRecord,
    ClauseReference,
    CorpusScope,
    MarkdownChunk,
    RetrievalHit,
)
from personal_knowledge.tools import (
    RetrievalToolRequest,
    build_retrieval_tool_registry,
)


def _chunk(
    chunk_id: str,
    line_start: int,
    text: str,
    *,
    specification: str = "24.501",
    heading_path: tuple[str, ...] = ("5.1 Registration procedure",),
) -> MarkdownChunk:
    return MarkdownChunk(
        chunk_id=chunk_id,
        source_collection="3gpp_markdown",
        specification=specification,
        version="17.1.0",
        source_path=f"data/3gpp/{specification}/v17.1.0/source.md",
        document_sha256="A" * 64,
        heading_path=heading_path,
        line_start=line_start,
        line_end=line_start + 1,
        text=text,
    )


def _record(
    clause_id: str,
    chunks: tuple[MarkdownChunk, ...],
    *,
    heading_path: tuple[str, ...],
    cross_references: tuple[ClauseReference, ...] = (),
) -> ClauseRecord:
    return ClauseRecord(
        specification="24.501",
        version="17.1.0",
        release=17,
        source_path="data/3gpp/24.501/v17.1.0/source.md",
        document_sha256="A" * 64,
        clause_id=clause_id,
        title=heading_path[-1].removeprefix(clause_id).strip(),
        heading_path=heading_path,
        heading_level=len(heading_path),
        line_start=min(item.line_start for item in chunks),
        direct_line_end=max(item.line_end for item in chunks),
        line_end=max(item.line_end for item in chunks),
        direct_chunk_ids=tuple(item.chunk_id for item in chunks),
        chunk_ids=tuple(item.chunk_id for item in chunks),
        cross_references=cross_references,
    )


class _Reranker:
    def predict(self, query, passages):
        return tuple(float(index) for index, _ in enumerate(passages, 1))


def _registry():
    first = _chunk("C1", 10, "The UE shall initiate registration.")
    second = _chunk("C2", 12, "The network shall process registration.")
    chunks = (first, second)
    clauses = (_record(
        "5.1",
        chunks,
        heading_path=first.heading_path,
        cross_references=(ClauseReference(
            raw_text="3GPP TS 24.501 clause 5.1",
            line_number=10,
            target_clause_id="5.1",
            target_specification="24.501",
        ),),
    ),)
    retriever = SimpleNamespace(
        corpus_scope=CorpusScope(release=17),
        chunks=chunks,
        chunk_by_id={item.chunk_id: item for item in chunks},
        clause_index=ClauseIndex(clauses, chunks),
        keyword_candidates=lambda query, **kwargs: [("C1", 3.0)],
        semantic_candidates=lambda query, **kwargs: [("C2", 0.1)],
        hybrid_hits=lambda request_id, query, **kwargs: ((RetrievalHit(
            hit_id="H1",
            chunk_id="C1",
            collection_rank=1,
            source_collection="3gpp_markdown",
            specification="24.501",
            version="17.1.0",
            source_path=first.source_path,
            document_sha256=first.document_sha256,
            heading_path=first.heading_path,
            line_start=first.line_start,
            line_end=first.line_end,
            excerpt=first.text,
            semantic_rank=1,
            semantic_distance=0.1,
            lexical_rank=1,
            lexical_score=3.0,
            rrf_score=0.03,
        ),), ()),
    )
    return build_retrieval_tool_registry(retriever, reranker=_Reranker())


def _request(**updates) -> RetrievalToolRequest:
    values = {
        "request_id": "REQ1",
        "query_id": "T1",
        "query": "registration",
        "limit": 4,
    }
    values.update(updates)
    return RetrievalToolRequest(**values)


def test_registry_restores_all_original_tools_and_descriptions():
    registry = _registry()

    assert registry.names == (
        "clause_lookup",
        "cross_reference_lookup",
        "expand_source_context",
        "hybrid_search",
        "hyde_search",
        "keyword_search",
        "navigate_clause",
        "rerank",
        "semantic_search",
    )
    assert all(item.description.strip() for item in registry.definitions)
    hybrid = next(item for item in registry.definitions if item.name == "hybrid_search")
    assert set(hybrid.parameters["properties"]) == {
        "query_id",
        "max_results",
        "target_specifications",
        "lexical_query",
        "semantic_query",
    }


def test_registry_rejects_invalid_native_tool_arguments():
    registry = _registry()

    try:
        registry.parse_call(NativeFunctionCall(
            call_id="C1",
            name="hybrid_search",
            arguments={"query_id": "T1", "max_results": 6},
        ))
    except ValueError as exc:
        assert "Field required" in str(exc)
    else:
        raise AssertionError("invalid arguments were accepted")


def test_cross_reference_requires_an_observed_source_edge():
    registry = _registry()

    resolved = registry.invoke("cross_reference_lookup", _request(
        query="",
        seed_chunk_ids=("C1",),
        reference_specification="24.501",
        reference_clause="5.1",
        limit=1,
    ))
    rejected = registry.invoke("cross_reference_lookup", _request(
        query="",
        seed_chunk_ids=("C1",),
        reference_specification="24.501",
        reference_clause="5.2",
    ))

    assert [item.chunk_id for item in resolved.evidence] == ["C1"]
    assert resolved.next_cursor == 1
    assert rejected.status == "rejected"


def test_keyword_hybrid_and_rerank_preserve_score_semantics():
    registry = _registry()

    keyword = registry.invoke("keyword_search", _request())
    hybrid = registry.invoke("hybrid_search", _request(
        semantic_query="registration procedure"
    ))
    reranked = registry.invoke("rerank", _request(
        query="registration procedure",
        semantic_query="registration procedure",
        seed_chunk_ids=("C1", "C2"),
        limit=1,
    ))

    assert keyword.evidence[0].score_kind == "bm25"
    assert hybrid.evidence[0].score_kind == "rrf"
    assert reranked.evidence[0].chunk_id == "C2"
    assert reranked.evidence[0].score_kind == "cross_encoder"
