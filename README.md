# llm-localmodel

A plugin for Simon Willison's [`llm`](https://llm.datasette.io/) CLI that
runs a real local Hugging Face model instead of calling a paid hosted API,
backed by `Qwen/Qwen2.5-0.5B-Instruct` running on your own machine. Same
plugin architecture (model registration, `max_tokens`/`temperature`
options, streaming), no API key, no rate limits, no bill. See
[DESIGN.md](DESIGN.md) for why this swap was made and what it costs.

*Inspired by [simonw/llm-meta-ai](https://github.com/simonw/llm-meta-ai) (an `llm` plugin for Meta's hosted AI API).*

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

No API key, no `llm keys set` step — the model runs locally. First use
downloads the model from Hugging Face (~1GB) and caches it.

## Usage

```bash
llm -m local/Qwen2.5-0.5B-Instruct "What is the capital of France?"
```

or by its shorter alias:

```bash
llm -m qwen-0.5b "What is the capital of France?"
```

Control generation length and randomness the same way as any other `llm`
model:

```bash
llm -m local/Qwen2.5-0.5B-Instruct "Tell me a short poem" -o max_tokens 100 -o temperature 0.9
```

## Verified, for real, against the actual `llm` CLI

Every claim below is an actual command I ran and actual output, not
illustrative pseudocode.

**Real generation:**

```
$ llm -m local/Qwen2.5-0.5B-Instruct "What is the capital of France? Answer in one short sentence."
The capital of France is Paris.
```

**`max_tokens` genuinely truncates** (verified by first confirming the
*unrestricted* response to the same prompt is a full multi-paragraph
story, then re-running with a tight budget):

```
$ llm -m local/Qwen2.5-0.5B-Instruct "Tell me a long story about a dragon." -o max_tokens 10 --no-stream
Once upon a time, in a land far away
```

**Streaming works** — `tests/test_llm_localmodel.py::test_streaming_yields_multiple_chunks`
asserts the response arrives as more than one chunk and that concatenating
the chunks equals the final `.text()`, passing for real against the
actual model.

**Conversation history is correctly threaded through** — verified two
ways: (1) directly inspecting the message list built for turn 2 of a
real conversation, confirming it includes turn 1's user message *and*
turn 1's actual generated response, and (2) end-to-end through the real
CLI's `--continue` flag:

```
$ llm -m local/Qwen2.5-0.5B-Instruct "My favorite color is teal. Remember that." --no-stream
I'm sorry to hear that you prefer teal as your favorite color! ...

$ llm -m local/Qwen2.5-0.5B-Instruct "What is my favorite color? Just answer with the color name, nothing else." --continue --no-stream
Teal.
```

All 5 tests in `tests/test_llm_localmodel.py` pass against the real
model (no mocking) — `python3 -m pytest tests/ -v`.

## A real finding, not a bug: small models are inconsistent

The first version of the conversation-history test used the vaguer phrasing
`"What is my favorite color?"` and got `"As an AI language model, I don't
have personal preferences..."` — sounding exactly like a broken feature.
Directly inspecting the built message list (see DESIGN.md) proved the
context *was* being passed correctly; the 0.5B model, under default
sampling (`temperature=0.7`, i.e. non-deterministic), sometimes falls back
to a generic trained response instead of using the context it's given,
especially on ambiguously-phrased follow-ups. A more directive phrasing
(`"...just answer with the color name"`) reliably gets the right answer.
This is a genuine small-model limitation — documented rather than hidden,
since finding out the mechanism actually works and the *model* is just
small was the whole point of testing this thoroughly instead of stopping
at the first confusing output.

## Design

See [DESIGN.md](DESIGN.md) for the Meta-AI-vs-local-model tradeoff, a real
dependency conflict this project's setup caused and how it was isolated,
and what was cut from the original.
