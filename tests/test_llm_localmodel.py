"""
Real tests against the actual local model -- no mocking. Downloads and
runs Qwen2.5-0.5B-Instruct (~1GB), so this is slower than a typical unit
test suite, but it's the only way to actually verify generation,
streaming, options, and conversation history work end-to-end.
"""

import llm


def get_model():
    return llm.get_model("local/Qwen2.5-0.5B-Instruct")


def test_model_registered():
    model = get_model()
    assert model.model_id == "local/Qwen2.5-0.5B-Instruct"
    assert model.can_stream is True


def test_basic_prompt():
    model = get_model()
    response = model.prompt("What is 2 + 2? Answer with just the number.", stream=False)
    text = response.text()
    assert "4" in text


def test_max_tokens_truncates():
    model = get_model()
    response = model.prompt(
        "Tell me a long story about a dragon.", stream=False, max_tokens=10
    )
    text = response.text()
    # a real long-story completion is normally 200+ tokens; bounded to 10 it
    # should come back short
    assert len(text.split()) < 20


def test_streaming_yields_multiple_chunks():
    model = get_model()
    response = model.prompt("Count from 1 to 5.", stream=True)
    chunks = list(response)
    assert len(chunks) > 1
    assert "".join(chunks) == response.text()


def test_conversation_history_is_used():
    model = get_model()
    conv = model.conversation()
    conv.prompt("My favorite color is teal. Remember that.", stream=False).text()
    r2 = conv.prompt(
        "What is my favorite color? Just answer with the color name, nothing else.",
        stream=False,
    )
    assert "teal" in r2.text().lower()
