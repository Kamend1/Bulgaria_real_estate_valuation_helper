"""
Regression test for a real incident (2026-09-10): a market-analyst turn
that kept calling tools for every one of max_iterations rounds used to
exit the for/else branch with final_text="" and NO on_message call at
all -- the appraiser saw a fully empty turn (no text, no error) after
several real, billed tool calls had already run. Fixed by forcing one
additional, tools-unbound "synthesis" call so the loop always produces
SOME text. See app/services/llm/orchestrator_graph.py::_build_tool_loop's
`else` branch.
"""
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from app.services.llm import orchestrator_graph as og


def _make_fake_get_chat_model(tool_call_rounds: int, synthesis_text: str):
    """Returns a fake get_chat_model whose bound (.bind_tools(...)) stream
    always requests a tool call, while the UNBOUND stream (the forced
    synthesis call, called directly on chat_model without bind_tools)
    returns real text -- mirrors exactly how _build_tool_loop uses both."""
    def fake_get_chat_model(provider, model_id, max_tokens=None, **kwargs):
        chat = MagicMock()

        def stream_no_tools(messages):
            chunk = AIMessageChunk(content=synthesis_text)
            chunk.usage_metadata = {"input_tokens": 500, "output_tokens": 40}
            yield chunk

        def stream_with_tools(messages):
            chunk = AIMessageChunk(content="", tool_calls=[{"id": "call", "name": "dummy_tool", "args": {}}])
            chunk.usage_metadata = {"input_tokens": 100, "output_tokens": 20}
            yield chunk

        chat.stream = stream_no_tools
        bound = MagicMock()
        bound.stream = stream_with_tools
        chat.bind_tools = lambda tools: bound
        return chat
    return fake_get_chat_model


def test_iteration_cap_forces_a_real_synthesis_instead_of_silence(monkeypatch):
    synthesis_text = "Обобщение на находките дотук."
    monkeypatch.setattr(og, "get_chat_model", _make_fake_get_chat_model(3, synthesis_text))

    dummy_tool = MagicMock()
    dummy_tool.name = "dummy_tool"
    dummy_tool.invoke = lambda args: {"ok": True}

    on_message_calls = []
    node = og._build_tool_loop(
        provider="anthropic", model_id="claude-opus-5", sampling_kwargs={}, max_tokens=2000,
        tools=[dummy_tool], system_prompt="test", max_iterations=3,
        call_log=(call_log := []), call_label_prefix="msg1_market_analysis",
        on_progress=lambda msg: None,
        on_message=lambda role, content, tool_calls, tool_call_id, truncated=False: on_message_calls.append((role, content, truncated)),
        step_label="Мисля", db=None, report_id=None, memory_source="chat", memory_source_id=None,
    )

    result = node({"messages": [], "findings": {}, "hops": 0})

    # The critical assertion: the turn must NOT end in silence.
    assert result["findings"]["market_analysis"] == synthesis_text
    assistant_texts = [content for role, content, _ in on_message_calls if role == "assistant"]
    assert synthesis_text in assistant_texts

    # The forced call is logged like any other step, for cost visibility.
    assert call_log[-1]["call_label"] == "msg1_market_analysis_synthesis"
    assert call_log[-1]["input_tokens"] == 500


def test_iteration_cap_synthesis_respects_truncation_flag(monkeypatch):
    # If even the forced synthesis call itself gets cut off by the token
    # cap, is_length_truncated must still be checked -- the fix shouldn't
    # trade "silent empty turn" for "silent truncated turn".
    def fake_get_chat_model(provider, model_id, max_tokens=None, **kwargs):
        chat = MagicMock()

        def stream_no_tools(messages):
            chunk = AIMessageChunk(content="частичен текст")
            chunk.usage_metadata = {"input_tokens": 500, "output_tokens": 2000}
            chunk.response_metadata = {"stop_reason": "max_tokens"}
            yield chunk

        def stream_with_tools(messages):
            chunk = AIMessageChunk(content="", tool_calls=[{"id": "call", "name": "dummy_tool", "args": {}}])
            chunk.usage_metadata = {"input_tokens": 100, "output_tokens": 20}
            yield chunk

        chat.stream = stream_no_tools
        bound = MagicMock()
        bound.stream = stream_with_tools
        chat.bind_tools = lambda tools: bound
        return chat

    monkeypatch.setattr(og, "get_chat_model", fake_get_chat_model)
    dummy_tool = MagicMock()
    dummy_tool.name = "dummy_tool"
    dummy_tool.invoke = lambda args: {"ok": True}

    on_message_calls = []
    node = og._build_tool_loop(
        provider="anthropic", model_id="claude-opus-5", sampling_kwargs={}, max_tokens=2000,
        tools=[dummy_tool], system_prompt="test", max_iterations=2,
        call_log=[], call_label_prefix="msg1_market_analysis",
        on_progress=lambda msg: None,
        on_message=lambda role, content, tool_calls, tool_call_id, truncated=False: on_message_calls.append((role, content, truncated)),
        step_label="Мисля", db=None, report_id=None, memory_source="chat", memory_source_id=None,
    )
    node({"messages": [], "findings": {}, "hops": 0})

    assistant_calls = [c for c in on_message_calls if c[0] == "assistant" and c[1]]
    assert assistant_calls[-1][2] is True   # truncated flag propagated
