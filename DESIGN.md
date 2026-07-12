# Design Doc: llm-localmodel

## Goal

Build a real, working plugin for Simon Willison's `llm` CLI, faithfully
implementing its actual plugin interface (`register_models` hook, a
`Model` subclass with `execute()`, an `Options` class, streaming support),
and verify it end-to-end against the real CLI and real generated text —
without needing a Meta AI API key, which this build has no access to.

## Why not the real Meta AI API

`llm-meta-ai` wraps a paid, authenticated hosted API. There's no free tier
to test against here, and — as with the LlamaParse situation in the
`tweetcharts` project — building an untested integration against an API
I can't call isn't verifiable work. The part worth reproducing faithfully
is the *plugin architecture itself*: how `llm` discovers a plugin, how a
`Model` registers, how options/streaming/conversation-history work. All of
that is real and fully testable with any backend, including a local one.

## Why Qwen2.5-0.5B-Instruct specifically

Needed a model that (a) has a proper instruction/chat template (so
multi-turn conversation and system prompts work the same way
`llm-meta-ai`'s reasoning models do), (b) is small enough to download and
run inference with on a laptop CPU/MPS in seconds, not minutes, and (c) is
a real, current, actively-maintained model rather than a toy/deprecated
one. Qwen2.5-0.5B-Instruct fits all three — verified generation quality
in the README is coherent, on-topic completions, not gibberish.

## A real dependency conflict, and how it was isolated

Installing `llm` globally pulled in `openai>=2.0`, which broke an
already-installed `langchain-openai` (needs `openai<2.0.0`) elsewhere on
this machine — a real, verified regression (confirmed via `pip check`
before/after), not a hypothetical risk. This is the same class of problem
as the `facenet-pytorch`-downgrades-torch incident in the `facerec`
project, and the `datasets`-bumps-protobuf incident in `build-nanogpt`.

Fix, and a change in approach: uninstalled `llm` from the global
environment, restored `openai<2.0.0` globally, and rebuilt this entire
project inside a **virtualenv created with `--system-site-packages`**
(`python3 -m venv --system-site-packages .venv`). This inherits the
already-installed heavy packages (`torch`, `transformers` — no
multi-gigabyte re-download) while isolating anything `pip install`'d
*inside* the venv (here: `llm` and its own `openai`/`pydantic` versions)
from ever touching the global site-packages. Verified directly: `python3
-c "import openai; print(openai.__file__)"` shows a different path (and a
different version, 2.45.0 vs the global 1.109.1) inside vs. outside the
venv.

This is a better fix than the previous projects' "install globally, then
manually patch the fallout" approach — it prevents the fallout in the
first place, for this project and (as a pattern worth reusing) any future
one that needs a package with aggressive/unpinned dependencies.

## What was verified, and how

- **Plugin discovery**: `llm plugins` and `llm models` show
  `llm-localmodel` and `local/Qwen2.5-0.5B-Instruct` — the entry-point
  registration in `pyproject.toml` actually works, not just looks correct.
- **Generation correctness**: real prompts, real (coherent, correct)
  answers, through the actual `llm` CLI subprocess, not a Python-level
  mock of it.
- **`max_tokens`**: proved genuine effect by first establishing the
  *unrestricted* baseline response length (a multi-paragraph story) before
  showing the restricted version cuts off mid-sentence — a restricted
  output alone wouldn't prove the option does anything, since a short
  response could just be what the model produced anyway.
- **Streaming**: asserted more than one chunk arrives *and* that
  concatenating them equals the non-streaming text — checks both "does
  streaming happen" and "is streamed output the same content as
  non-streamed," which are different failure modes.
- **Conversation history**: verified at two levels — direct inspection of
  the constructed message list (proves `_build_messages` is correct) and
  end-to-end through the real CLI's `--continue` flag (proves `llm`'s own
  conversation-reconstruction-from-SQLite-log mechanism correctly feeds
  into this plugin). The first end-to-end attempt got a wrong-looking
  answer; testing at both levels was what made it possible to tell
  "context is being passed but the model doesn't use it well here" apart
  from "the plugin's context-passing is broken" — see the README's
  "real finding, not a bug" section.

## What was cut

- **No API key / auth handling.** The whole point of a local model is not
  needing one; `needs_key = None` reflects that honestly rather than
  keeping a vestigial, always-empty key-handling code path.
- **No `llm meta-ai models` / `llm meta-ai refresh` style model-listing
  commands.** There's one fixed local model here, not a fleet of hosted
  ones behind an API that can add/remove models server-side, so a
  refreshable model list doesn't apply.
- **No attachments (images/PDFs) or tool-calling support.** Qwen2.5-0.5B
  is text-only and this plugin doesn't implement `attachment_types` or
  `supports_tools`. A larger, multimodal local model could add these
  following the same `Model` interface, but wasn't necessary to prove the
  plugin architecture itself works.
- **No `reasoning_effort` option.** That's specific to reasoning models
  that expose a controllable thinking budget; Qwen2.5-0.5B-Instruct isn't
  one, so exposing the option would do nothing and be dishonest about
  what the model actually supports.
