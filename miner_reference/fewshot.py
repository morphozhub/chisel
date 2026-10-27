"""Committed few-shot examples and procedural patterns appended to the instructions.

Everything here is read from ``library/`` inside the committed source tree at
import time, so the exact instruction text is a pure function of the admitted
commit (replay sees the same bytes). Files are read in a fixed, sorted order.
"""

from __future__ import annotations

import json
from pathlib import Path

_DIR = Path(__file__).with_name("library")


def _load() -> str:
    if not _DIR.is_dir():
        return ""
    parts: list[str] = []
    patterns = _DIR / "patterns.md"
    if patterns.is_file():
        parts.append("## Procedural patterns (allowlisted APIs only)\n\n" + patterns.read_text().strip())
    index = _DIR / "index.json"
    names: list[str]
    if index.is_file():
        names = [str(row["file"]) for row in json.loads(index.read_text())]
    else:
        names = sorted(p.name for p in _DIR.glob("*.js"))
    examples = []
    for name in names:
        path = _DIR / name
        if path.is_file():
            examples.append(f"```javascript\n{path.read_text().strip()}\n```")
    if examples:
        parts.append(
            "## Reference examples of valid, well-structured modules (different objects; "
            "imitate the construction style, not the object)\n\n" + "\n\n".join(examples)
        )
    return "\n\n".join(parts)


LIBRARY = _load()
