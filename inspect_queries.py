#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def main() -> None:
    src = Path("kg_queries.jsonl")
    out = Path("kg_queries.pretty.txt")
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")

    text = src.read_text(encoding="utf-8-sig")
    data: List[Dict[str, object]]
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
            query_id = rec.get("query_id", "")
            w.write(f"=== {i} | {kg_id} | {query_id} ===\n")
            w.write(str(rec.get("sparql_clean", "")).strip() + "\n\n")

    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
