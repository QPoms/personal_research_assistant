"""Framework-independent native tool-calling contracts from SpecRAG."""

from .tool_calling import (
    NativeFunctionCall,
    NativeFunctionCallOutput,
    NativeFunctionDefinition,
    NativeToolCapability,
)

__all__ = [
    "NativeFunctionCall",
    "NativeFunctionCallOutput",
    "NativeFunctionDefinition",
    "NativeToolCapability",
]
