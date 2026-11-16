"""Publish PyLLaMa to the Hugging Face Hub.

By default the LoRA adapter folder is uploaded as-is (a few dozen MB; no model
is loaded, so it needs no GPU). With ``--merge`` the adapter is merged into the
base weights and the full ~13 GB model is uploaded instead.
Log in first with ``huggingface-cli login`` (or ``notebook_login()`` in Colab).

Usage::

    python -m pyllama.push_to_hub --adapter outputs/pyllama-lora/final --repo-id <hf-user>/PyLLaMa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyllama import DEFAULT_BASE

MODEL_CARD = """---
base_model: {base}
library_name: peft
license: llama2
tags: [code, python, lora, qlora, codellama]
---

# PyLLaMa

[{base}](https://huggingface.co/{base}) fine-tuned with LoRA (QLoRA, 4-bit) on an
instruction dataset built from the official Python documentation (docs.python.org).

Source code, dataset builder and benchmark: https://github.com/geraldketu/Pyllama

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base}", device_map="auto", dtype=torch.float16)
model = PeftModel.from_pretrained(base, "{repo}")
tokenizer = AutoTokenizer.from_pretrained("{repo}")

prompt = "[INST] How do I sort a list of dicts by a key? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=256)[0], skip_special_tokens=True))
```

Or, from the GitHub repo: `python -m pyllama.generate --adapter {repo} "your question"`.
{training}"""


def model_card(base: str, repo: str, training_notes: str | None) -> str:
    training = f"\n## Training\n\n{training_notes}\n" if training_notes else ""
    return MODEL_CARD.format(base=base, repo=repo, training=training)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--merge", action="store_true", help="upload merged full weights instead of the adapter")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--training-notes", help="markdown added to the model card's Training section")
    args = ap.parse_args(argv)

    if not (args.adapter / "adapter_config.json").exists():
        sys.exit(f"No LoRA adapter at {args.adapter} (adapter_config.json missing). Train first.")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)

    if args.merge:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from pyllama.compat import dtype_kwarg

        base = AutoModelForCausalLM.from_pretrained(args.base_model, **dtype_kwarg(torch.float16))
        merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
        merged.push_to_hub(args.repo_id, private=args.private)
        AutoTokenizer.from_pretrained(str(args.adapter)).push_to_hub(args.repo_id, private=args.private)
    else:
        api.upload_folder(
            folder_path=str(args.adapter),
            repo_id=args.repo_id,
            ignore_patterns=["checkpoint-*", "*.pt", "optimizer*", "scheduler*", "rng_state*", "trainer_state.json"],
            commit_message="Upload PyLLaMa LoRA adapter",
        )

    api.upload_file(
        path_or_fileobj=model_card(args.base_model, args.repo_id, args.training_notes).encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        commit_message="Add model card",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
