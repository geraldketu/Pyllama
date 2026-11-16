"""LoRA fine-tuning of CodeLlama on the Python-docs instruction dataset.

Loss is computed on the answer tokens only (prompt tokens are masked with -100).
Pass ``--load-in-4bit`` for QLoRA so a 7B model fits on a single 16 GB GPU
(e.g. a free Colab T4).

Usage::

    python -m pyllama.train --train data/processed/train.json --val data/processed/val.json \
        --output-dir outputs/pyllama-lora --load-in-4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from pyllama.prompts import build_example

DEFAULT_BASE = "codellama/CodeLlama-7b-Instruct-hf"


def tokenize_example(rec: dict, tokenizer, max_len: int) -> dict:
    prompt, completion = build_example(rec["instruction"], rec["output"], rec.get("input", ""))
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = [tokenizer.bos_token_id] + prompt_ids + completion_ids
    labels = [-100] * (1 + len(prompt_ids)) + completion_ids
    input_ids, labels = input_ids[:max_len], labels[:max_len]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def load_split(path: Path, tokenizer, max_len: int) -> Dataset:
    rows = json.loads(path.read_text(encoding="utf-8"))
    ds = Dataset.from_list(rows)
    ds = ds.map(lambda r: tokenize_example(r, tokenizer, max_len), remove_columns=ds.column_names)
    # Drop examples whose answer was entirely truncated away.
    return ds.filter(lambda r: any(label != -100 for label in r["labels"]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--train", type=Path, default=Path("data/processed/train.json"))
    ap.add_argument("--val", type=Path, default=Path("data/processed/val.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/pyllama-lora"))
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA (requires bitsandbytes + CUDA)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    model_kwargs = {"torch_dtype": compute_dtype, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = load_split(args.train, tokenizer, args.max_len)
    val_ds = load_split(args.val, tokenizer, args.max_len) if args.val.exists() else None
    print(f"train examples: {len(train_ds)}, val examples: {len(val_ds) if val_ds else 0}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=val_ds is not None,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    final_dir = args.output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"LoRA adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
