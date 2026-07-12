"""
llm plugin backed by a local Hugging Face model instead of a paid hosted
API. Same idea as llm-meta-ai (register a Model, handle options, support
streaming) minus the API key -- see DESIGN.md for why this swap was made
and what it costs.
"""

from threading import Thread
from typing import Optional

import llm
from pydantic import Field

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_pipeline_cache = {}


def _get_model_and_tokenizer(hf_model_id):
    if hf_model_id not in _pipeline_cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(hf_model_id, torch_dtype="auto").to(device)
        _pipeline_cache[hf_model_id] = (model, tokenizer, device)
    return _pipeline_cache[hf_model_id]


@llm.hookimpl
def register_models(register):
    register(LocalModel(DEFAULT_MODEL_ID), aliases=("qwen-0.5b",))


class LocalModel(llm.Model):
    can_stream = True
    needs_key = None  # no API key -- this is the whole point

    class Options(llm.Options):
        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate", default=512
        )
        temperature: Optional[float] = Field(
            description="Sampling temperature", default=0.7
        )

    def __init__(self, hf_model_id):
        self.hf_model_id = hf_model_id
        self.model_id = f"local/{hf_model_id.split('/')[-1]}"

    def _build_messages(self, prompt, conversation):
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        if conversation:
            for response in conversation.responses:
                messages.append({"role": "user", "content": response.prompt.prompt})
                messages.append({"role": "assistant", "content": response.text_or_raise()})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt, stream, response, conversation):
        import torch
        from transformers import TextIteratorStreamer

        model, tokenizer, device = _get_model_and_tokenizer(self.hf_model_id)
        messages = self._build_messages(prompt, conversation)

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        attention_mask = torch.ones_like(input_ids)

        max_new_tokens = prompt.options.max_tokens or 512
        temperature = prompt.options.temperature
        do_sample = bool(temperature and temperature > 0)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

        if stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            thread = Thread(target=model.generate, kwargs={**gen_kwargs, "streamer": streamer})
            thread.start()
            for token_text in streamer:
                yield token_text
            thread.join()
        else:
            with torch.no_grad():
                output_ids = model.generate(**gen_kwargs)
            new_tokens = output_ids[0][input_ids.shape[1]:]
            yield tokenizer.decode(new_tokens, skip_special_tokens=True)
