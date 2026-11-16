"""Prompt formatting shared by training, inference and benchmarking.

CodeLlama-Instruct uses the Llama-2 chat format: ``<s>[INST] ... [/INST] answer </s>``.
Keeping the template in one place guarantees the model sees the same format at
train time and at inference time.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are PyLLaMa, an expert Python assistant. Answer accurately and concisely, "
    "following the official Python documentation. When code is useful, return it in a "
    "single ```python block."
)


def build_prompt(instruction: str, input_text: str = "", system: str = SYSTEM_PROMPT) -> str:
    """Return the prompt part (everything up to and including ``[/INST]``).

    The BOS token is not included; the tokenizer adds it.
    """
    user = instruction.strip()
    if input_text and input_text.strip():
        user = f"{user}\n\n{input_text.strip()}"
    return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


def build_example(instruction: str, output: str, input_text: str = "") -> tuple[str, str]:
    """Return ``(prompt, completion)`` for a training example."""
    return build_prompt(instruction, input_text), " " + output.strip()
