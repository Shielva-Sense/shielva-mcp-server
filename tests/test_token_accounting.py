"""Tokens must survive the trip from the provider to the analytics row.

🚨 ARC's per-bot Analytics showed 0 tokens and $0.000 cost for every bot while
the workspace-level meter read 316.8K. Two independent faults, both here:

1. `routing/llm_router.py` set `tokens_used` from the LAST tool-loop response
   only — the exact mistake the metering comment directly above it was written
   to warn about, made one line further down. A tool-using turn spends a full
   completion each time round.
2. Nothing carried the input/output SPLIT, so a caller could only ever apply a
   blended rate. Providers price read and written tokens differently, often by
   an order of magnitude.
"""

from __future__ import annotations


def test_the_routing_loop_sums_every_iteration_not_just_the_last() -> None:
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "src/routing/llm_router.py").read_text()

    assert "spent_total += " in src, "the loop does not accumulate across iterations"
    assert "tokens_used=spent_total" in src, "the response still reports one iteration"
    # The truncation path spent the MOST — five completions and no answer.
    assert "tokens_used=0" not in src, "hitting the iteration cap still reports zero tokens"


def test_every_hop_carries_the_input_output_split() -> None:
    """A total alone forces a blended rate. Each hop between the provider and
    core-api must pass the split along, or the split is lost at the narrowest
    one and every hop after it is guessing."""
    from src.application.chat.handle_query import HandleQueryOutput
    from src.application.llm.tool_loop import ToolLoopOutput
    from src.interface.http.llm_router import LLMCompleteResponse
    from src.protocol.models import MCPQueryResponse
    from src.routing.llm_router import LLMResponse

    for carrier in (LLMResponse, ToolLoopOutput, HandleQueryOutput):
        fields = getattr(carrier, "__dataclass_fields__", {})
        assert "input_tokens" in fields, f"{carrier.__name__} drops the split"
        assert "output_tokens" in fields, f"{carrier.__name__} drops the split"

    for model in (LLMCompleteResponse, MCPQueryResponse):
        assert "input_tokens" in model.model_fields, f"{model.__name__} drops the split"
        assert "output_tokens" in model.model_fields, f"{model.__name__} drops the split"


def test_the_split_is_reported_from_provider_usage_not_invented() -> None:
    """`LLMUsage` has carried prompt/completion the whole time — the numbers were
    there and were being thrown away at the last step."""
    from src.domain.llm.value_objects import LLMUsage

    f = LLMUsage.__dataclass_fields__
    assert {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(f)

    # Read from disk: `src.interface.http.llm_router` resolves to the APIRouter
    # INSTANCE of that name, not the module, so inspect cannot reach the source.
    import pathlib

    router_src = (pathlib.Path(__file__).resolve().parents[1] / "src/interface/http/llm_router.py").read_text()
    assert "input_tokens=response.usage.prompt_tokens" in router_src
    assert "output_tokens=response.usage.completion_tokens" in router_src
