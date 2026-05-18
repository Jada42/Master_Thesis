"""JSONL and JSON loading helpers used across the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return its entries as a list of dicts.

    Args:
        file_path: Path to the ``.jsonl`` file.

    Returns:
        Each line parsed as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)
    entries: list[dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_json(data: Any, file_path: str | Path) -> None:
    """Save data as a pretty-printed JSON file.

    Args:
        data: Any JSON-serializable Python object.
        file_path: Destination path (will be overwritten).
    """
    file_path = Path(file_path)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
