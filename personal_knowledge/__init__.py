"""Embedded private-knowledge retrieval derived from the SpecRAG core.

Only corpus discovery, hybrid retrieval, clause provenance, and source
validation are retained. Authentication, workspaces, HTTP/MCP, frontend,
Docker, memory, context compaction, and worker lifecycle are intentionally not
part of this package.
"""

from .service import LocalPersonalKnowledgeService

__all__ = ["LocalPersonalKnowledgeService"]

