"""Small file helpers shared by the loop, scorer and reflection cycle."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def dump_yaml(path: Path, data: dict) -> None:
    atomic_write(path, yaml.safe_dump(data, sort_keys=False))


def write_json(path: Path, data: Any) -> None:
    atomic_write(path, json.dumps(data, indent=2))


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
