# PyLLaMa

**CodeLlama fine-tuned with LoRA on the official Python documentation.**

PyLLaMa is a domain-specialized code assistant. The pipeline scrapes
[docs.python.org](https://docs.python.org/3/), turns it into an instruction-tuning
dataset, fine-tunes `CodeLlama-7b-Instruct` with LoRA/QLoRA, publishes the model to
the Hugging Face Hub, and benchmarks it against ChatGPT-3.5 on NeetCode-style
problems by measuring CodeBERT embedding similarity to reference solutions.

```
docs.python.org ──scrape──▶ raw pages (JSONL) ──build──▶ train/val JSON
                                                            │
                                         LoRA fine-tune ◀───┘
                                               │
                     Hugging Face Hub ◀──push──┤
                                               ▼
   NeetCode-style problems ──▶ PyLLaMa / GPT-3.5 answers ──CodeBERT──▶ cosine similarity
```

## Repository layout

| Path | Purpose |
|---|---|
| `pyllama/scrape.py` | Polite BFS crawler (robots.txt, rate limit, retries, on-disk cache) that parses Sphinx pages into sections, prose, code examples and API entries (`dl.py` blocks) |
| `pyllama/build_dataset.py` | Converts pages into Alpaca-style `{instruction, input, output}` records (API, concept and code tasks), de-duplicates, splits train/val **by page** |
| `pyllama/prompts.py` | Single source of truth for the CodeLlama-Instruct `[INST]` prompt template |
| `pyllama/train.py` | LoRA fine-tuning with PEFT + 🤗 Trainer; loss on answer tokens only; optional 4-bit QLoRA |
| `pyllama/generate.py` | Load base model + adapter and generate answers |
| `pyllama/push_to_hub.py` | Upload the adapter (or merged weights) and a model card to the Hugging Face Hub |
| `pyllama/benchmark.py` | `generate` answers with PyLLaMa or OpenAI models; `evaluate` with CodeBERT cosine similarity |
| `benchmark/problems.json` | 20 NeetCode-150-style problems across 12 categories with reference solutions |
| `tests/` | Unit tests for parsing, dataset building, prompt format, scoring, and correctness of every reference solution |

## Quickstart

```bash
pip install -r requirements.txt
```

### 1. Build the dataset

```bash
python -m pyllama.scrape --out data/raw/python_docs.jsonl --max-pages 400
python -m pyllama.build_dataset --raw data/raw/python_docs.jsonl --out-dir data/processed
```

Each record looks like:

```json
{
  "instruction": "How do I use `int.bit_length` in Python?",
  "input": "",
  "output": "`int.bit_length()`\n\nReturn the number of bits necessary to represent an integer in binary, excluding the sign and leading zeros: ...",
  "source": "https://docs.python.org/3/library/stdtypes.html#int.bit_length",
  "task": "api"
}
```

A 12-page sample crawl (Built-in Types, collections, built-in functions, data model, ...)
yields ~1,000 examples. The default 400-page crawl covers the library reference,
tutorial, HOWTOs and language reference.

### 2. Fine-tune

**Easiest:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/geraldketu/Pyllama/blob/main/colab/train_pyllama.ipynb)
runs the whole pipeline (dataset → training → Hugging Face upload → benchmark) on a Colab GPU.
Everything is saved to Google Drive and **Run all** is safe to repeat: finished steps are skipped and
interrupted training resumes from its last checkpoint.

A 7B model with QLoRA fits on a single 16 GB GPU (e.g. Colab T4/L4):

```bash
python -m pyllama.train --load-in-4bit --output-dir outputs/pyllama-lora
```

Defaults: LoRA `r=16`, `alpha=32`, dropout `0.05` on `q/k/v/o_proj`, lr `2e-4`
(cosine), 3 epochs, effective batch size 16, max length 1024.

### 3. Try it and publish

```bash
python -m pyllama.generate --adapter outputs/pyllama-lora/final --load-in-4bit \
    "How do I count word frequencies with collections.Counter?"

huggingface-cli login
python -m pyllama.push_to_hub --adapter outputs/pyllama-lora/final --repo-id <hf-user>/PyLLaMa
```

### 4. Benchmark against ChatGPT-3.5

```bash
python -m pyllama.benchmark generate --backend pyllama --adapter outputs/pyllama-lora/final --load-in-4bit --name pyllama
OPENAI_API_KEY=... python -m pyllama.benchmark generate --backend openai --model gpt-3.5-turbo --name gpt-3.5-turbo
python -m pyllama.benchmark evaluate results/pyllama.jsonl results/gpt-3.5-turbo.jsonl
```

`evaluate` extracts the code from each answer, embeds it and the reference solution
with [CodeBERT](https://huggingface.co/microsoft/codebert-base) (mean-pooled last
hidden state), and writes per-problem cosine similarity, per-model means and a
head-to-head win count to `results/summary.md`.

> **Caveat:** embedding similarity measures how close a solution is to the
> reference in style and structure, not whether it is functionally correct. It is
> a cheap proxy that complements, rather than replaces, unit-test-based metrics
> such as pass@k.

## Results so far

**Training run (Colab, NVIDIA L4, Nov 2026)**

| Setting | Value |
|---|---|
| Data | 150 docs.python.org pages → 5,509 examples (4,132 API, 844 concept, 533 code) |
| Split | 5,038 train / 471 validation, split by page |
| Method | QLoRA: 4-bit NF4 base, LoRA r=16, α=32 on q/k/v/o (16.8M trainable params, 0.25%) |
| Schedule | 1 epoch, 315 steps, effective batch 16, lr 2e-4 cosine, max length 768 |
| Time | ~40 minutes |
| Loss | train 1.55 at step 10, 1.23 averaged over the epoch · **validation 1.226** |

Validation loss was 1.229 at 63% of the epoch and 1.226 at the end. That's a small gain from the
second half, so more epochs or more pages are the next lever.

**Benchmark:** PyLLaMa answers for all 20 problems are generated. The GPT-3.5 comparison and
CodeBERT scoring are still to be run (section 7 of the notebook); results will be added here.

## Tests

```bash
pytest -q
```

## License

Code: MIT. Fine-tuned weights inherit the [Llama 2 Community License](https://ai.meta.com/llama/license/)
of the CodeLlama base model. Python documentation content © Python Software
Foundation, used under the [PSF License](https://docs.python.org/3/license.html).
