"""Small synchronous boundary around the embedded SpecRAG retrieval core."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from threading import Lock
from typing import Any

from .config import RetrievalCfg, knowledge_root
from .retrieval.hybrid import HybridMarkdownRetriever
from .retrieval.models import RetrievalHit
from .tools import ToolEvidence, ToolRegistry, build_retrieval_tool_registry


@dataclass(slots=True)
class LocalPersonalKnowledgeService:
    cfg: RetrievalCfg
    max_results: int = 6
    allow_index_build: bool = False
    _retriever: HybridMarkdownRetriever | None = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    def _get_retriever(self) -> HybridMarkdownRetriever:
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    self._retriever = HybridMarkdownRetriever(
                        self.cfg,
                        allow_index_build=self.allow_index_build,
                    )
        return self._retriever

    def build_tool_registry(self) -> ToolRegistry:
        """Expose SpecRAG's original in-process retrieval tool surface."""

        return build_retrieval_tool_registry(self._get_retriever())

    def validate_tool_evidence(self, evidence: ToolEvidence) -> dict[str, Any]:
        """Revalidate a tool result against the immutable source on disk."""

        item = self._validated_evidence(evidence)
        item.update({
            "retrieval_tool": evidence.match_kind,
            "score": evidence.score,
            "score_kind": evidence.score_kind,
            "chunk_id": evidence.chunk_id,
        })
        return item

    def search(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        request_id = "PK-" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:12].upper()
        retriever = self._get_retriever()
        hits, errors = retriever.hybrid_hits(
            request_id,
            query,
            limit=self.max_results,
        )
        return self._result(query, hits, errors, retrieval_tool="hybrid_search")

    @staticmethod
    def _evidence_id(chunk_id: str) -> str:
        return "E-" + hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:16].upper()

    def _result(
        self,
        query: str,
        hits: tuple[RetrievalHit, ...],
        errors: tuple[str, ...],
        *,
        retrieval_tool: str,
    ) -> dict[str, Any]:
        evidence = [self._validated_evidence(hit) for hit in hits]
        valid = [item for item in evidence if item["source_valid"]]
        status = "completed" if valid and not errors else "partial" if valid else "unavailable"
        return {
            "status": status,
            "query": query,
            "retrieval_tool": retrieval_tool,
            "release": self.cfg.target_release,
            "answer": None,
            "evidence": valid,
            "evidence_status": "retrieved" if valid else "insufficient",
            "unresolved_gaps": list(errors),
            "failure": None if valid else {"category": "no_valid_evidence", "message": "; ".join(errors)},
        }

    @staticmethod
    def _validated_evidence(hit: Any) -> dict[str, Any]:
        root = knowledge_root()
        source = (root / hit.source_path).resolve()
        path_allowed = source.is_relative_to(root) and source.is_file()
        hash_valid = False
        locator_valid = False
        excerpt_valid = False
        if path_allowed:
            raw = source.read_bytes()
            hash_valid = hashlib.sha256(raw).hexdigest().upper() == hit.document_sha256
            lines = source.read_text(encoding="utf-8").splitlines()
            locator_valid = 1 <= hit.line_start <= hit.line_end <= len(lines)
            if locator_valid:
                located = "\n".join(lines[hit.line_start - 1 : hit.line_end])
                excerpt_valid = hit.excerpt.strip() in located
        source_valid = path_allowed and hash_valid and locator_valid and excerpt_valid
        heading = " > ".join(hit.heading_path)
        locator_parts = [
            f"Release {hit.release}" if hit.release else "",
            hit.version or "",
            heading,
            f"Lines {hit.line_start}-{hit.line_end}",
        ]
        return {
            "evidence_id": LocalPersonalKnowledgeService._evidence_id(hit.chunk_id),
            "source_type": "personal",
            "title": f"3GPP TS {hit.specification}",
            "specification": hit.specification,
            "release": hit.release,
            "version": hit.version,
            "locator": " / ".join(part for part in locator_parts if part),
            "heading_path": list(hit.heading_path),
            "line_start": hit.line_start,
            "line_end": hit.line_end,
            "excerpt": hit.excerpt,
            "source_path": hit.source_path,
            "document_sha256": hit.document_sha256,
            "semantic_rank": getattr(hit, "semantic_rank", None),
            "lexical_rank": getattr(hit, "lexical_rank", None),
            "rrf_score": getattr(hit, "rrf_score", 0.0),
            "source_valid": source_valid,
        }

