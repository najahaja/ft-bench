"""Artifact I/O, JSONL helpers, and HuggingFace Hub utilities."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def read_jsonl(path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict[str, Any]], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def push_to_hub(local_path: str, repo_id: str, path_in_repo: str = "") -> None:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        if os.path.isdir(local_path):
            api.upload_folder(folder_path=local_path, repo_id=repo_id,
                              path_in_repo=path_in_repo, repo_type="model")
        else:
            api.upload_file(path_or_fileobj=local_path,
                            path_in_repo=path_in_repo or os.path.basename(local_path),
                            repo_id=repo_id, repo_type="model")
        print(f"[OK] Pushed {local_path} -> hub:{repo_id}/{path_in_repo}")
    except Exception as e:
        print(f"[!] HF Hub push failed: {e}")
