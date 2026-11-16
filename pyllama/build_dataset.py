"""Turn scraped documentation pages into an instruction-tuning dataset (JSON).

Every record has the Alpaca-style fields ``instruction``, ``input``, ``output``
plus ``source`` (doc URL) and ``task`` (which template produced it). Three kinds
of examples are generated:

* ``api``      - "What does `str.split` do?" -> signature, description, example
* ``concept``  - "Explain <section> (<page>)" -> prose + example code
* ``code``     - "Write Python code that demonstrates <section>" -> code example

Usage::

    python -m pyllama.build_dataset --raw data/raw/python_docs.jsonl --out-dir data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

MIN_OUTPUT_CHARS = {"api": 80, "concept": 80, "code": 40}
MAX_OUTPUT_CHARS = 4000
MAX_CODE_BLOCKS = 2

API_TEMPLATES = (
    "What does `{name}` do in Python?",
    "Explain the Python {kind} `{name}` and show how to use it.",
    "How do I use `{name}` in Python?",
)
CONCEPT_TEMPLATES = (
    "Explain {title} in Python.",
    "Describe {title} as covered in the Python documentation for {page}.",
)
CODE_TEMPLATE = "Write Python code that demonstrates {title}."


def _fence(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    cut = text[:MAX_OUTPUT_CHARS]
    # Never leave an unterminated code fence.
    if cut.count("```") % 2 == 1:
        cut = cut[: cut.rfind("```")]
    return cut.rstrip()


def _pick(templates: tuple[str, ...], key: str) -> str:
    """Deterministically choose a template so rebuilding the dataset is reproducible."""
    return templates[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(templates)]


def api_examples(page: dict) -> list[dict]:
    out = []
    for e in page["api_entries"]:
        parts = [f"`{e['signature']}`", e["description"]]
        if e["code"]:
            parts.append("Example:\n" + "\n\n".join(_fence(c) for c in e["code"][:MAX_CODE_BLOCKS]))
        out.append(
            {
                "instruction": _pick(API_TEMPLATES, e["name"]).format(name=e["name"], kind=e["kind"]),
                "input": "",
                "output": _truncate("\n\n".join(parts)),
                "source": f"{page['url']}#{e['name']}",
                "task": "api",
            }
        )
    return out


def section_examples(page: dict) -> list[dict]:
    out = []
    for s in page["sections"]:
        title = s["title"].strip()
        source = f"{page['url']}#{s['anchor']}" if s["anchor"] else page["url"]
        if s["paragraphs"]:
            body = "\n\n".join(s["paragraphs"])
            if s["code"]:
                body += "\n\n" + "\n\n".join(_fence(c) for c in s["code"][:MAX_CODE_BLOCKS])
            out.append(
                {
                    "instruction": (
                        CONCEPT_TEMPLATES[0] if title == page["title"] else _pick(CONCEPT_TEMPLATES, source)
                    ).format(title=title, page=page["title"]),
                    "input": "",
                    "output": _truncate(body),
                    "source": source,
                    "task": "concept",
                }
            )
        for i, code in enumerate(s["code"][:MAX_CODE_BLOCKS]):
            out.append(
                {
                    "instruction": CODE_TEMPLATE.format(title=title),
                    "input": "",
                    "output": _truncate(_fence(code)),
                    "source": f"{source}::code{i}",
                    "task": "code",
                }
            )
    return out


def build(pages: list[dict]) -> list[dict]:
    records, seen = [], set()
    for page in pages:
        for rec in api_examples(page) + section_examples(page):
            if len(rec["output"]) < MIN_OUTPUT_CHARS[rec["task"]]:
                continue
            key = (rec["instruction"], rec["output"])
            if key in seen:
                continue
            seen.add(key)
            records.append(rec)
    return records


def split(records: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split by documentation page so validation pages are unseen during training."""
    pages = sorted({r["source"].split("#")[0] for r in records})
    random.Random(seed).shuffle(pages)
    n_val = max(1, int(len(pages) * val_frac)) if len(pages) > 1 else 0
    val_pages = set(pages[:n_val])
    train = [r for r in records if r["source"].split("#")[0] not in val_pages]
    val = [r for r in records if r["source"].split("#")[0] in val_pages]
    return train, val


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/python_docs.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    with args.raw.open(encoding="utf-8") as f:
        pages = [json.loads(line) for line in f if line.strip()]
    records = build(pages)
    train, val = split(records, args.val_frac, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.json", train), ("val.json", val)):
        (args.out_dir / name).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    by_task = {}
    for r in records:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    print(f"{len(records)} examples from {len(pages)} pages {by_task}")
    print(f"train={len(train)} val={len(val)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
