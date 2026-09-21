"""
Dataset acquisition, cleaning, deduplication, and JSONL export for MASSIVE.

Uses the OFFICIAL train/validation/test splits from AmazonScience/massive.
Does NOT re-split. Performs a leakage check and logs results to data/README.md.
"""
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

from ftbench.common.io import write_jsonl, write_json
from ftbench.common.seed import seed_everything


def fingerprint(text: str) -> str:
    """MD5 fingerprint of lowercased + whitespace-normalized text for deduplication."""
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def _parse_annot_utt(annot_utt: str) -> dict[str, str]:
    """
    Parse MASSIVE bracket-annotated utterance into a slots dict.
    Format: "I want to [slot_type : value] and more text"
    Returns {slot_type: verbatim_value, ...}
    """
    import re
    slots = {}
    for m in re.finditer(r"\[([^:]+)\s*:\s*([^\]]+)\]", annot_utt):
        slot_type = m.group(1).strip()
        slot_value = m.group(2).strip()
        slots[slot_type] = slot_value
    return slots


def build_label_vocab(dataset) -> dict:
    """Extract intent and slot type vocabularies from the MASSIVE dataset."""
    intent_feat = dataset["train"].features.get("intent")
    if hasattr(intent_feat, "names") and intent_feat.names:
        intents = sorted(intent_feat.names)
    else:
        intents = sorted(set(dataset["train"]["intent"]))

    all_annots = (
        list(dataset["train"]["annot_utt"])
        + list(dataset["validation"]["annot_utt"])
        + list(dataset["test"]["annot_utt"])
    )
    slot_types = sorted(set(
        slot_type
        for annot in all_annots
        for slot_type in _parse_annot_utt(annot).keys()
    ))
    return {"intents": intents, "slots": slot_types}


def load_massive(config: str = "en-US"):
    """Download MASSIVE from HF Hub, return DatasetDict with official splits."""
    from datasets import load_dataset
    print(f"[+] Downloading AmazonScience/massive ({config})...")
    try:
        return load_dataset("AmazonScience/massive", config, trust_remote_code=True)
    except Exception as e:
        print(f"[+] Direct script loading failed ({e}). Loading official HF parquet conversion...")
        base_url = f"https://huggingface.co/datasets/AmazonScience/massive/resolve/refs%2Fconvert%2Fparquet/{config}"
        return load_dataset(
            "parquet",
            data_files={
                "train": f"{base_url}/train/0000.parquet",
                "validation": f"{base_url}/validation/0000.parquet",
                "test": f"{base_url}/test/0000.parquet",
            },
        )


def _record_from_row(row: dict, split_name: str, idx: int, intent_names: Optional[List[str]] = None) -> dict:
    """Convert a MASSIVE row to an FT-Bench record."""
    from ftbench.prompts.templates import build_inference_prompt
    utterance = row.get("utt", "")
    annot_utt = row.get("annot_utt", "")
    intent_val = row.get("intent", "")
    if isinstance(intent_val, int) and intent_names and 0 <= intent_val < len(intent_names):
        intent = intent_names[intent_val]
    else:
        intent = str(intent_val)
    slots = _parse_annot_utt(annot_utt)
    ground_truth = {"intent": intent, "slots": slots}
    prompt = build_inference_prompt(utterance)
    return {
        "id": f"{split_name}_{idx:06d}",
        "utterance": utterance,
        "ground_truth": ground_truth,
        "prompt": prompt,
        "fingerprint": fingerprint(utterance),
    }


def _check_leakage(
    train_records: List[dict],
    val_records: List[dict],
    test_records: List[dict],
) -> Tuple[List[dict], int]:
    """
    Remove train rows whose fingerprint appears in val or test.
    Returns (clean_train, n_removed).
    """
    val_fps = {r["fingerprint"] for r in val_records}
    test_fps = {r["fingerprint"] for r in test_records}
    leaked = val_fps | test_fps
    clean = [r for r in train_records if r["fingerprint"] not in leaked]
    return clean, len(train_records) - len(clean)


def prepare(
    output_dir: str = "data",
    config: str = "en-US",
    seed: int = 42,
    use_synthetic: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, str]:
    """
    Full pipeline: download -> process -> leakage-check -> save.
    Returns dict with paths: {"train": ..., "val": ..., "test": ...}
    """
    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)

    if use_synthetic:
        print("WARNING: SYNTHETIC DATA — NOT FOR REAL RESULTS")
        records = _generate_synthetic(max_samples or 100)
        write_jsonl(records, os.path.join(output_dir, "train.jsonl"))
        write_jsonl([], os.path.join(output_dir, "val.jsonl"))
        write_jsonl([], os.path.join(output_dir, "test.jsonl"))
        paths = {
            "train": os.path.join(output_dir, "train.jsonl"),
            "val": os.path.join(output_dir, "val.jsonl"),
            "test": os.path.join(output_dir, "test.jsonl"),
        }
        _write_readme(output_dir, len(records), 0, 0, 0)
        return paths

    dataset = load_massive(config)

    # Build and persist vocabulary
    vocab = build_label_vocab(dataset)
    write_json(vocab, "configs/label_vocab.json")
    print(f"[+] Vocabulary: {len(vocab['intents'])} intents, {len(vocab['slots'])} slot types")

    intent_feat = dataset["train"].features.get("intent")
    intent_names = intent_feat.names if hasattr(intent_feat, "names") else None

    # Convert splits
    train_records = [_record_from_row(row, "train", i, intent_names) for i, row in enumerate(dataset["train"])]
    val_records   = [_record_from_row(row, "val",   i, intent_names) for i, row in enumerate(dataset["validation"])]
    test_records  = [_record_from_row(row, "test",  i, intent_names) for i, row in enumerate(dataset["test"])]

    # Leakage check
    train_records, n_leaked = _check_leakage(train_records, val_records, test_records)
    print(f"[+] Leakage check: removed {n_leaked} train rows with fingerprints in val/test")

    # Apply max_samples if set (for smoke tests only)
    if max_samples:
        train_records = train_records[:max_samples]

    paths = {}
    for split_name, split_data in [("train", train_records), ("val", val_records), ("test", test_records)]:
        p = os.path.join(output_dir, f"{split_name}.jsonl")
        write_jsonl(split_data, p)
        paths[split_name] = p
        print(f"[+] Wrote {len(split_data)} records -> {p}")

    _write_readme(output_dir, len(train_records), len(val_records), len(test_records), n_leaked)
    print("[OK] Dataset preparation complete.")
    return paths


def _write_readme(output_dir, n_train, n_val, n_test, n_leaked):
    readme = (
        f"train: {n_train} rows | val: {n_val} rows | test: {n_test} rows\n"
        f"leakage_removed_from_train: {n_leaked} rows\n"
    )
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)


# ── Synthetic fallback (CPU smoke test, no network, no GPU) ───────────────────
_SYNTHETIC_INTENTS = [
    "alarm_set", "audio_volume_down", "calendar_query", "datetime_query",
    "email_sendemail", "general_quirky", "iot_hue_lighton", "music_query",
    "news_query", "play_music", "qa_factoid", "recommendation_movies",
    "takeaway_order", "transport_query", "weather_query",
]

_SYNTHETIC_UTTERANCES = [
    "set an alarm for 7am tomorrow",
    "turn the volume down please",
    "what is on my calendar today",
    "what time is it in tokyo",
    "send an email to john",
    "tell me a joke",
    "turn on the lights",
    "play some jazz music",
    "what is in the news today",
    "play bohemian rhapsody",
    "who invented the telephone",
    "recommend a good movie",
    "order me a pizza",
    "how do i get to the airport",
    "will it rain tomorrow",
]


def _generate_synthetic(n: int = 100) -> list:
    """Generate synthetic NLUParse-shaped records for offline smoke testing."""
    from ftbench.prompts.templates import build_inference_prompt
    records = []
    for i in range(n):
        intent = _SYNTHETIC_INTENTS[i % len(_SYNTHETIC_INTENTS)]
        utt = _SYNTHETIC_UTTERANCES[i % len(_SYNTHETIC_UTTERANCES)]
        gt = {"intent": intent, "slots": {}}
        records.append({
            "id": f"synth_{i:05d}",
            "utterance": utt,
            "ground_truth": gt,
            "prompt": build_inference_prompt(utt),
            "fingerprint": fingerprint(utt + str(i)),
        })
    return records
