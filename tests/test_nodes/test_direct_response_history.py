"""Conversation context survives the graph's direct-response shortcut."""

from langchain_core.messages import AIMessage, HumanMessage

from isaac.nodes.direct_response import _build_direct_prompt


def test_direct_prompt_includes_prior_exchange() -> None:
    history = [HumanMessage(content="My name is Ada"), AIMessage(content="Hello Ada")]
    prompt = _build_direct_prompt("What is my name?", "", history)
    assert [message.content for message in prompt[-3:]] == [
        "My name is Ada",
        "Hello Ada",
        "What is my name?",
    ]
