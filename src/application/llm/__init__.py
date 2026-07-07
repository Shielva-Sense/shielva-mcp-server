"""LLM application services.

* :class:`LLMApplicationService` — single-turn completion wrapper
  with audit (used by codegen + any future single-shot caller).
* :class:`CompleteWithToolLoopUseCase` — multi-turn tool-calling
  loop composing LLMProvider + ToolCatalogue + ToolExecutor. The
  generic loop fix-agent migrates onto in slice 4c.
"""

from .services import LLMApplicationService
from .tool_loop import (
    CompleteWithToolLoopUseCase,
    ExecutedToolCall,
    ToolLoopInput,
    ToolLoopOutput,
)

__all__ = [
    "CompleteWithToolLoopUseCase",
    "ExecutedToolCall",
    "LLMApplicationService",
    "ToolLoopInput",
    "ToolLoopOutput",
]
