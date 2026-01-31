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

    def format_sparql(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        formatted: List[str] = []
        indent = 0
        for line in lines:
            lowered = line.lower()
            if lowered.startswith("}"):
                indent = max(indent - 1, 0)
            formatted.append("  " * indent + line)
            if line.endswith("{"):
                indent += 1
        return "\n".join(formatted)

    with out.open("w", encoding="utf-8") as w:
        for i, rec in enumerate(data, 1):
            kg_id = rec.get("kg_id", "")
            query_id = rec.get("query_id", "")
            query_label = rec.get("query_label", "")
            w.write(f"=== {i} | {kg_id} | {query_label} | {query_id} ===\n")
            w.write(json.dumps(rec, ensure_ascii=False, indent=2))
            sparql = rec.get("sparql_clean")
            if isinstance(sparql, str) and sparql.strip():
                w.write("\n\nSPARQL (formatted)\n")
                w.write(format_sparql(sparql) + "\n")
            w.write("\n\n")

    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
