"""NeetCode-style benchmark: generate solutions, then score them against reference answers.

Two sub-commands:

``generate`` - ask a model to solve every problem in ``benchmark/problems.json`` and
write ``results/<name>.jsonl``. Supported backends: ``pyllama`` (local base +
LoRA adapter) and ``openai`` (e.g. gpt-3.5-turbo; needs ``OPENAI_API_KEY``).

``evaluate`` - embed each generated solution and the reference solution with
CodeBERT (``microsoft/codebert-base``, mean pooling) and report cosine similarity
per problem, per model, and a head-to-head win count.

Usage::

    python -m pyllama.benchmark generate --backend pyllama --adapter outputs/pyllama-lora/final --name pyllama
    python -m pyllama.benchmark generate --backend openai --model gpt-3.5-turbo --name gpt-3.5-turbo
    python -m pyllama.benchmark evaluate results/pyllama.jsonl results/gpt-3.5-turbo.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROBLEMS = Path(__file__).resolve().parent.parent / "benchmark" / "problems.json"
INSTRUCTION = "Solve the following problem in Python. Return only the code in a single ```python block.\n\n{prompt}"
FENCE_RE = re.compile(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def load_problems(path: Path = PROBLEMS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_code(text: str) -> str:
    """Return the code from a model answer: the longest fenced block, or the raw text."""
    blocks = FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def cosine(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


# --------------------------------------------------------------------------- generate


def _openai_answer(client, model: str, instruction: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": instruction}],
    )
    return resp.choices[0].message.content or ""


def generate(args: argparse.Namespace) -> None:
    problems = load_problems(args.problems)
    if args.backend == "pyllama":
        from pyllama.generate import PyLLaMa

        model = PyLLaMa(args.base_model, args.adapter, args.load_in_4bit)
        answer = lambda instr: model.generate(instr, max_new_tokens=args.max_new_tokens)  # noqa: E731
    else:
        from openai import OpenAI

        client = OpenAI()
        answer = lambda instr: _openai_answer(client, args.model, instr)  # noqa: E731

    out = Path("results") / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, p in enumerate(problems, 1):
            raw = answer(INSTRUCTION.format(prompt=p["prompt"]))
            f.write(json.dumps({"id": p["id"], "raw": raw, "code": extract_code(raw)}) + "\n")
            print(f"[{i}/{len(problems)}] {p['id']}")
    print(f"-> {out}")


# --------------------------------------------------------------------------- evaluate


class CodeBERTEmbedder:
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def __call__(self, code: str):
        torch = self.torch
        enc = self.tokenizer(code, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.model(**enc).last_hidden_state  # (1, seq, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return ((hidden * mask).sum(1) / mask.sum(1)).squeeze(0).numpy()


def evaluate(args: argparse.Namespace) -> None:
    problems = {p["id"]: p for p in load_problems(args.problems)}
    embed = CodeBERTEmbedder(args.embedder)
    ref_vecs = {pid: embed(p["reference"]) for pid, p in problems.items()}

    scores: dict[str, dict[str, float]] = {}
    for path in args.results:
        name = Path(path).stem
        scores[name] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in ref_vecs:
                scores[name][row["id"]] = cosine(embed(row["code"]), ref_vecs[row["id"]])

    names = list(scores)
    ids = [pid for pid in problems if all(pid in scores[n] for n in names)]
    lines = ["| Problem | Difficulty | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for pid in ids:
        cells = " | ".join(f"{scores[n][pid]:.4f}" for n in names)
        lines.append(f"| {problems[pid]['title']} | {problems[pid]['difficulty']} | {cells} |")
    means = " | ".join(f"**{sum(scores[n][p] for p in ids) / len(ids):.4f}**" for n in names)
    lines.append(f"| **Mean** | | {means} |")

    if len(names) == 2:
        a, b = names
        wins = sum(scores[a][p] > scores[b][p] for p in ids)
        lines.append("")
        lines.append(f"{a} closer to the reference than {b} on {wins}/{len(ids)} problems.")

    report = "\n".join(lines)
    print(report)
    out = Path("results") / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# CodeBERT cosine similarity to reference solutions\n\n{report}\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"-> {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", type=Path, default=PROBLEMS)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--backend", choices=["pyllama", "openai"], required=True)
    g.add_argument("--name", required=True, help="results/<name>.jsonl")
    g.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model (openai backend)")
    g.add_argument("--base-model", default="codellama/CodeLlama-7b-Instruct-hf")
    g.add_argument("--adapter", default=None)
    g.add_argument("--load-in-4bit", action="store_true")
    g.add_argument("--max-new-tokens", type=int, default=512)
    g.set_defaults(func=generate)

    e = sub.add_parser("evaluate")
    e.add_argument("results", nargs="+", help="results/*.jsonl files to compare")
    e.add_argument("--embedder", default="microsoft/codebert-base")
    e.set_defaults(func=evaluate)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
