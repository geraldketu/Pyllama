"""Load PyLLaMa (base model + LoRA adapter, or a merged checkpoint) and generate answers.

Usage::

    python -m pyllama.generate --adapter outputs/pyllama-lora/final \
        "How do I read a CSV file with the csv module?"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pyllama.compat import dtype_kwarg
from pyllama.prompts import build_prompt
from pyllama import DEFAULT_BASE


class PyLLaMa:
    def __init__(self, base_model: str = DEFAULT_BASE, adapter: str | None = None, load_in_4bit: bool = False):
        if adapter and not Path(adapter).is_dir() and not re.fullmatch(r"[\w.-]+/[\w.-]+", adapter):
            # Otherwise a missing local folder surfaces as a confusing "Repo id must be in the form..." error.
            raise SystemExit(f"No adapter found at {adapter}. Train first, or pass a Hugging Face repo id like user/PyLLaMa.")
        kwargs = {**dtype_kwarg(torch.float16 if torch.cuda.is_available() else torch.float32), "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, instruction: str, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        prompt = build_prompt(instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if temperature > 0 else {"do_sample": False}
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **sampling,
        )
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instruction")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=None, help="local path or Hugging Face repo id of the LoRA adapter")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args(argv)

    model = PyLLaMa(args.base_model, args.adapter, args.load_in_4bit)
    print(model.generate(args.instruction, args.max_new_tokens))


if __name__ == "__main__":
    main()
