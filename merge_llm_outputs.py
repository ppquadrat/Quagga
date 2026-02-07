#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    stripped = raw.lstrip("\ufeff").lstrip()
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return records
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    records.append(item)
            return records
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge llm_outputs.jsonl into kg_queries.jsonl.")
    parser.add_argument("--queries", default="kg_queries.jsonl")
    parser.add_argument("--llm", default="llm_outputs.jsonl")
    parser.add_argument("--out", default="kg_queries.jsonl")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    llm_path = Path(args.llm)
    out_path = Path(args.out)

    queries = load_jsonl(queries_path)
    llm_records = load_jsonl(llm_path)

    by_query_id: Dict[str, Dict[str, object]] = {}
    for rec in llm_records:
        qid = rec.get("query_id")
        if isinstance(qid, str) and qid:
            by_query_id[qid] = rec

    updated = 0
    missing = 0
    for rec in queries:
        if not isinstance(rec, dict):
            missing += 1
            continue
        qid = rec.get("query_id")
        if not isinstance(qid, str):
            missing += 1
            continue
        llm_rec = by_query_id.get(qid)
        if not llm_rec:
            missing += 1
            continue
        llm_output = llm_rec.get("llm_output")
        if isinstance(llm_output, dict):
            rec["llm_output"] = llm_output
            if isinstance(llm_rec.get("model"), str):
                rec["llm_model"] = llm_rec.get("model")
            if isinstance(llm_rec.get("generated_at"), str):
                rec["llm_generated_at"] = llm_rec.get("generated_at")
            if isinstance(llm_rec.get("elapsed_ms"), int):
                rec["llm_elapsed_ms"] = llm_rec.get("elapsed_ms")
            updated += 1

    with out_path.open("w", encoding="utf-8") as f:
        for rec in queries:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Updated {updated} records; {missing} without LLM output. Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
