"""
Prompt construction for FT-Bench NLU task.

All prompts are built via tokenizer.apply_chat_template().
There are NO hand-written Llama3/ChatML template string literals here.
"""
from typing import Optional


SYSTEM_PROMPT = (
    "You are an NLU parser. Given an utterance, output a JSON object with exactly "
    "two keys:\n"
    "  \"intent\": one label from the provided vocabulary\n"
    "  \"slots\": object mapping slot type names to verbatim substrings of the input\n\n"
    "Output only the raw JSON. No markdown fences, no explanation.\n"
    "Vocabulary will be provided in the user turn."
)


def _vocab_block(label_vocab: Optional[dict] = None) -> str:
    """Format vocabulary section of the user turn."""
    if label_vocab is None:
        import json
        from pathlib import Path
        vocab_path = Path(__file__).resolve().parents[2] / "configs" / "label_vocab.json"
        with open(vocab_path) as f:
            label_vocab = json.load(f)
    intents_str = ", ".join(label_vocab.get("intents", []))
    slots_str = ", ".join(label_vocab.get("slots", []))
    return f"Intents: {intents_str}\nSlot types: {slots_str}"


def build_inference_prompt(
    utterance: str,
    tokenizer=None,
    label_vocab: Optional[dict] = None,
) -> str:
    """
    Build a formatted prompt string for a single utterance.

    If tokenizer is provided, uses tokenizer.apply_chat_template().
    Otherwise returns a plain-text representation for offline use.
    """
    vocab_section = _vocab_block(label_vocab)
    user_content = f"Vocabulary:\n{vocab_section}\n\nUtterance: {utterance}"

    if tokenizer is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Offline fallback — plain text only (for unit tests / prepare_dataset.py)
    return f"[SYSTEM]\n{SYSTEM_PROMPT}\n[USER]\n{user_content}\n[ASSISTANT]\n"


def build_training_prompt(
    utterance: str,
    ground_truth: dict,
    tokenizer=None,
    label_vocab: Optional[dict] = None,
) -> str:
    """Build a complete training example with the ground-truth assistant turn."""
    import json as _json
    vocab_section = _vocab_block(label_vocab)
    user_content = f"Vocabulary:\n{vocab_section}\n\nUtterance: {utterance}"
    assistant_content = _json.dumps(ground_truth)

    if tokenizer is not None:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n"
        f"[USER]\n{user_content}\n"
        f"[ASSISTANT]\n{assistant_content}"
    )
