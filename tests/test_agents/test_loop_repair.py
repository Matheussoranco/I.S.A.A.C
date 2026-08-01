"""AgentLoop integration: recovering malformed tool calls mid-run.

These cover the regression that motivated WS3 — a model emitting a tool call as
text used to end the run, with the raw JSON handed back as the final answer.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from isaac.agents.agent_loop import AgentLoop
from isaac.tools.base import IsaacTool, ToolResult


class SearchTool(IsaacTool):
    name = "web_search"
    description = "Search the web."
    risk_level = 1
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(success=True, output=f"RESULTS:{kwargs.get('query', '')}")


class ScriptedLLM:
    """Replays scripted turns and records what it was asked."""

    def __init__(self, turns: list[AIMessage]) -> None:
        self._turns = turns
        self._i = 0
        self.seen: list[list[Any]] = []
        self.bound_schemas: Any = None
        self.bound_kwargs: dict[str, Any] = {}

    def bind_tools(self, schemas: Any) -> ScriptedLLM:
        self.bound_schemas = schemas
        return self

    def bind(self, **kwargs: Any) -> ScriptedLLM:
        self.bound_kwargs = kwargs
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.seen.append(list(messages))
        turn = self._turns[min(self._i, len(self._turns) - 1)]
        self._i += 1
        return turn


class ScriptedOllama(ScriptedLLM):
    """A ScriptedLLM the constraint detector recognises as an Ollama client."""

    base_url = "http://localhost:11434"


class TestRepairInLoop:
    def test_fenced_json_call_is_executed_not_returned_as_the_answer(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                AIMessage(
                    content='```json\n{"name": "web_search", '
                    '"arguments": {"query": "chollet"}}\n```'
                ),
                AIMessage(content="Chollet works on ARC."),
            ]
        )
        result = AgentLoop([tool], llm=llm, auto_approve=True).run("who is chollet")

        assert tool.calls == [{"query": "chollet"}], "the salvaged call must actually run"
        assert result.output == "Chollet works on ARC."
        assert result.health.repaired == 1
        assert result.health.malformed_rate == 1.0

    def test_baseline_behaviour_is_restored_when_repair_is_disabled(self) -> None:
        tool = SearchTool()
        blob = '{"name": "web_search", "arguments": {"query": "x"}}'
        llm = ScriptedLLM([AIMessage(content=blob)])
        result = AgentLoop([tool], llm=llm, auto_approve=True, repair_tool_calls=False).run("q")

        # This is the 1.3.x failure the release fixes: no tool ran and the raw
        # blob came back as the answer.
        assert tool.calls == []
        assert result.output == blob

    def test_hermes_tag_dialect(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                AIMessage(
                    content='<tool_call>{"name": "web_search", '
                    '"arguments": {"query": "arc"}}</tool_call>'
                ),
                AIMessage(content="done"),
            ]
        )
        AgentLoop([tool], llm=llm, auto_approve=True).run("q")
        assert tool.calls == [{"query": "arc"}]

    def test_genuine_final_answer_still_ends_the_run(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM([AIMessage(content="Paris is the capital of France.")])
        result = AgentLoop([tool], llm=llm, auto_approve=True).run("capital?")

        assert tool.calls == []
        assert result.output == "Paris is the capital of France."
        assert result.stopped_reason == "final"
        assert result.health.malformed == 0

    def test_native_calls_are_not_counted_as_malformed(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": {"query": "x"},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        result = AgentLoop([tool], llm=llm, auto_approve=True).run("q")
        assert result.health.native == 1
        assert result.health.malformed == 0
        assert result.health.malformed_rate == 0.0


class TestReflexion:
    def test_unparseable_call_triggers_a_corrective_retry(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                AIMessage(content='I will call the web_search tool with {"query": '),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": {"query": "fixed"},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        result = AgentLoop([tool], llm=llm, auto_approve=True).run("q")

        assert tool.calls == [{"query": "fixed"}]
        assert result.health.reflexion_attempts == 1
        assert result.health.reflexion_recovered == 1
        # The retry must not also be counted as a clean native turn.
        assert result.health.native == 0

    def test_retry_message_names_the_available_tools(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                AIMessage(content='I will call the web_search tool with {"query": '),
                AIMessage(content="giving up"),
            ]
        )
        AgentLoop([tool], llm=llm, auto_approve=True).run("q")

        retry_turn = llm.seen[1]
        correction = retry_turn[-1]
        assert isinstance(correction, HumanMessage)
        assert "web_search" in correction.content

    def test_retries_are_budgeted(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM([AIMessage(content="I will call the web_search tool with {")])
        result = AgentLoop(
            [tool], llm=llm, auto_approve=True, reflexion_retries=1, max_iterations=8
        ).run("q")

        assert result.health.reflexion_attempts == 1, "must not retry forever"
        assert result.health.unrecovered == 1

    def test_reflexion_can_be_disabled(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM([AIMessage(content="I will call the web_search tool with {")])
        result = AgentLoop([tool], llm=llm, auto_approve=True, reflexion_retries=0).run("q")
        assert result.health.reflexion_attempts == 0


class TestObservationChannel:
    def test_repaired_call_results_go_back_as_plain_messages(self) -> None:
        # A ToolMessage whose tool_call_id matches no call on the assistant
        # turn is rejected by strict OpenAI-compatible servers, so a salvaged
        # call must report its result as an ordinary observation instead.
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                AIMessage(content='{"name": "web_search", "arguments": {"query": "x"}}'),
                AIMessage(content="done"),
            ]
        )
        AgentLoop([tool], llm=llm, auto_approve=True).run("q")

        followup = llm.seen[1]
        assert not any(isinstance(m, ToolMessage) for m in followup)
        assert any(isinstance(m, HumanMessage) and "RESULTS:x" in str(m.content) for m in followup)

    def test_native_call_results_use_tool_messages(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": {"query": "x"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        AgentLoop([tool], llm=llm, auto_approve=True).run("q")

        followup = llm.seen[1]
        tool_msgs = [m for m in followup if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "call_1"


class TestConstrainedMode:
    def test_envelope_call_is_executed(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                AIMessage(content='{"tool": "web_search", "arguments": {"query": "e"}}'),
                AIMessage(content='{"tool": "none", "final_answer": "all done"}'),
            ]
        )
        result = AgentLoop([tool], llm=llm, auto_approve=True, constrained_decoding=True).run("q")

        assert tool.calls == [{"query": "e"}]
        assert result.output == "all done"

    def test_constraint_is_bound_instead_of_native_tools(self) -> None:
        tool = SearchTool()
        # base_url makes this look like Ollama to the channel detector, which
        # is what selects the JSON-Schema `format` constraint.
        llm = ScriptedOllama([AIMessage(content='{"tool": "none", "final_answer": "hi"}')])
        AgentLoop([tool], llm=llm, auto_approve=True, constrained_decoding=True).run("q")

        assert llm.bound_schemas is None, "native tool binding must be bypassed"
        assert "oneOf" in llm.bound_kwargs["format"]

    def test_unconstrainable_provider_degrades_to_prompt_only(self, caplog) -> None:
        # No constraint channel: the envelope is still requested and parsed,
        # but the caller is warned that it is not enforced.
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                AIMessage(content='{"tool": "web_search", "arguments": {"query": "q"}}'),
                AIMessage(content='{"tool": "none", "final_answer": "ok"}'),
            ]
        )
        with caplog.at_level("WARNING"):
            result = AgentLoop([tool], llm=llm, auto_approve=True, constrained_decoding=True).run(
                "q"
            )

        assert tool.calls == [{"query": "q"}], "envelope must still be parsed"
        assert result.output == "ok"
        assert llm.bound_kwargs == {}
        assert any("prompt-only envelope" in r.message for r in caplog.records)

    def test_tool_catalogue_is_added_to_the_system_prompt(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM([AIMessage(content='{"tool": "none", "final_answer": "hi"}')])
        AgentLoop([tool], llm=llm, auto_approve=True, constrained_decoding=True).run("q")

        system = str(llm.seen[0][0].content)
        assert "web_search" in system
        assert "final_answer" in system


class TestHealthMetrics:
    def test_rates_are_zero_for_an_empty_run(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM([AIMessage(content="answer")])
        health = AgentLoop([tool], llm=llm, auto_approve=True).run("q").health
        assert health.malformed_rate == 0.0
        assert health.recovered_rate == 0.0

    def test_reflexion_credit_does_not_rob_an_earlier_repaired_turn(self) -> None:
        # Regression: the recovery credit used to be taken from whichever
        # cumulative counter was non-zero, so an *earlier* repaired turn paid
        # for a *later* reflexion recovery. That moved a malformed turn into
        # the native bucket and understated the headline malformed_rate.
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                # turn 1: salvaged from text -> repaired
                AIMessage(content='{"name": "web_search", "arguments": {"query": "a"}}'),
                # turn 2: unparseable attempt -> Reflexion retry
                AIMessage(content='I will call the web_search tool with {"query": '),
                # turn 3: the correction lands as a native call
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": {"query": "fixed"},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        health = AgentLoop([tool], llm=llm, auto_approve=True).run("q").health

        assert health.repaired == 1, "the first turn was repaired and must stay repaired"
        assert health.reflexion_recovered == 1
        assert health.native == 0, "no turn arrived cleanly through the native channel"
        assert health.malformed == 2
        assert health.malformed_rate == 1.0

    def test_as_dict_is_serialisable(self) -> None:
        tool = SearchTool()
        llm = ScriptedLLM(
            [
                AIMessage(content='{"name": "web_search", "arguments": {"query": "x"}}'),
                AIMessage(content="done"),
            ]
        )
        data = AgentLoop([tool], llm=llm, auto_approve=True).run("q").health.as_dict()
        assert data["repaired"] == 1
        assert data["malformed_rate"] == 1.0
        assert data["recovered_rate"] == 1.0
