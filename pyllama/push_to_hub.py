"""Publish PyLLaMa to the Hugging Face Hub.

By default the LoRA adapter is uploaded (small, a few MB). With ``--merge`` the
adapter is merged into the base weights and the full model is uploaded instead.
Log in first with ``huggingface-cli login``.

Usage::

    python -m pyllama.push_to_hub --adapter outputs/pyllama-lora/final --repo-id <hf-user>/PyLLaMa
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pyllama.train import DEFAULT_BASE

MODEL_CARD = """---
base_model: {base}
library_name: peft
license: llama2
tags: [code, python, lora, codellama]
---

# PyLLaMa

{base} fine-tuned with LoRA on an instruction dataset built from the official
Python documentation (docs.python.org). Source code: https://github.com/geraldketu/Pyllama

```python
from pyllama.generate import PyLLaMa
model = PyLLaMa(adapter="{repo}")
print(model.generate("How do I sort a list of dicts by a key?"))
```
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--merge", action="store_true", help="upload merged full weights instead of the adapter")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.adapter)
    if args.merge:
        model = model.merge_and_unload()

    model.push_to_hub(args.repo_id, private=args.private)
    tokenizer.push_to_hub(args.repo_id, private=args.private)

    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=MODEL_CARD.format(base=args.base_model, repo=args.repo_id).encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
