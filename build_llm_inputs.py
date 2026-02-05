#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


SPARQL_BLOCK_EVIDENCE_TYPES = {
    "repo_file",
    "md_pre",
    "md_fence",
    "doc_pre",
    "doc_fence",
    "doc_pdf",
}


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_evidence(
    evidence: Iterable[object],
    include_sparql_blocks: bool,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        ev_type = ev.get("type")
        snippet = ev.get("snippet")
        evidence_id = ev.get("evidence_id")
        if not isinstance(ev_type, str) or not isinstance(snippet, str) or not isinstance(evidence_id, str):
            continue
        if not snippet.strip():
            continue
        if not include_sparql_blocks and ev_type in SPARQL_BLOCK_EVIDENCE_TYPES:
            continue
        out.append(
            {
                "evidence_id": evidence_id,
                "type": ev_type,
                "snippet": snippet,
                "source_path": ev.get("source_path", ""),
                "source_url": ev.get("source_url", ""),
            }
        )
    return out


def build_prompt_input(
    rec: Dict[str, object],
    include_raw: bool,
    include_sparql_blocks: bool,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "query_id": rec.get("query_id"),
        "query_label": rec.get("query_label"),
        "kg_id": rec.get("kg_id"),
        "sparql_clean": rec.get("sparql_clean"),
        "evidence": iter_evidence(rec.get("evidence", []) or [], include_sparql_blocks),
        "schema_ref": "schemas/llm_output.schema.json",
    }
    if include_raw:
        payload["sparql_raw"] = rec.get("sparql_raw")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM prompt inputs from kg_queries.jsonl.")
    parser.add_argument(
        "--input",
        default="kg_queries.jsonl",
        help="Input query records JSONL (default: kg_queries.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="llm_inputs.jsonl",
        help="Output LLM input JSONL (default: llm_inputs.jsonl)",
    )
    parser.add_argument(
        "--include-raw-sparql",
        action="store_true",
        help="Include sparql_raw in each payload.",
    )
    parser.add_argument(
        "--include-sparql-evidence",
        action="store_true",
        help="Keep evidence types that are likely full SPARQL blocks (repo_file/md_pre/doc_pre/etc).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    records = load_jsonl(in_path)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            payload = build_prompt_input(
                rec,
                include_raw=args.include_raw_sparql,
                include_sparql_blocks=args.include_sparql_evidence,
            )
            if not isinstance(payload.get("query_id"), str) or not isinstance(payload.get("kg_id"), str):
                continue
            if not isinstance(payload.get("sparql_clean"), str) or not payload["sparql_clean"].strip():
                continue
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} prompt payloads to {out_path.resolve()}")


if __name__ == "__main__":
    main()
