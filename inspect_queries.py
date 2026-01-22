#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    src = Path("raw_queries.jsonl")
    out = Path("raw_queries.pretty.txt")
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")

    text = src.read_text(encoding="utf-8-sig")
    data: List[Dict[str, str]]
    if text.lstrip().startswith("["):
        data = json.loads(text)
    else:
        data = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    with out.open("w", encoding="utf-8") as w:
        for i, rec in enumerate(data, 1):
            kg_id = rec.get("kg_id", "")
            path = rec.get("path", "")
            w.write(f"=== {i} | {kg_id} | {path} ===\n")
            w.write(rec.get("query", "").strip() + "\n\n")

    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
