#!/usr/bin/env python3
from __future__ import annotations

"""
Extract SPARQL queries from Git repositories listed in `seeds.yaml`:
1) Read `seeds.yaml` for KG IDs and repo URLs.
2) Clone repos into `repos/` if missing, and record current commit hash.
3) Walk repo files and extract queries from `.rq`, `.sparql`, and fenced
   ```sparql blocks in Markdown.
4) Normalize queries, keep only SELECT queries, and deduplicate by sha256.
5) Write `kg_queries.jsonl` (one JSON object per query, with provenance).
"""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import yaml


@dataclass
class KGSeed:
    kg_id: str
    repos: List[str]


def load_seeds(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing seeds file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("seeds.yaml must contain a top-level mapping (dictionary).")
    kgs = data.get("kgs")
    if not isinstance(kgs, list):
        raise ValueError("seeds.yaml must have a top-level key 'kgs' containing a list.")
    for i, item in enumerate(kgs):
        if not isinstance(item, dict):
            raise ValueError(f"seeds.yaml: kgs[{i}] must be a mapping (dict).")
    return kgs


def parse_kg_seed(raw: Dict[str, object]) -> KGSeed:
    kg_id = raw.get("kg_id")
    if not isinstance(kg_id, str) or not kg_id.strip():
        raise ValueError("Each KG must have a non-empty string 'kg_id'.")
    repos = raw.get("repos") or []
    if not isinstance(repos, list) or not all(isinstance(x, str) for x in repos):
        raise ValueError(f"KG '{kg_id}': 'repos' must be a list of strings.")
    return KGSeed(kg_id=kg_id.strip(), repos=repos)


def repo_dir_from_url(repo_url: str) -> Path:
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Bad repo URL: {repo_url}")
    owner, repo = parts[-2], parts[-1]
    return Path(f"{owner}__{repo}")


def ensure_repo_cloned(repo_url: str, base_dir: Path) -> Path:
    repo_dir = base_dir / repo_dir_from_url(repo_url)
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", repo_url, str(repo_dir)],
            check=True,
        )
    return repo_dir


def get_repo_commit(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_dir,
    )
    return result.stdout.strip()


def iter_repo_files(repo_dir: Path) -> Iterable[Path]:
    for path in repo_dir.rglob("*"):
        if path.is_file():
            yield path


def normalize_query(text: str) -> str:
    normalized = text.strip()
    while normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    return normalized


def split_queries(text: str) -> List[str]:
    lines = text.splitlines()
    keyword_re = re.compile(
        r"^\s*(select|construct|ask|describe|insert|delete|with|load|clear|create|drop|copy|move|add)\b",
        re.IGNORECASE,
    )
    meta_re = re.compile(r"^\s*(prefix|base)\b", re.IGNORECASE)

    def is_meta_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if stripped.startswith("#"):
            return True
        return bool(meta_re.match(stripped))

    def strip_line_comments(line: str) -> str:
        # Only strip full-line comments; avoid breaking IRIs like http://...
        if line.lstrip().startswith("#"):
            return ""
        return line

    starts: List[int] = []
    depth = 0
    for idx, line in enumerate(lines):
        if depth == 0 and keyword_re.match(line):
            starts.append(idx)
        clean = strip_line_comments(line)
        depth += clean.count("{") - clean.count("}")

    if len(starts) <= 1:
        return [text]

    adjusted: List[int] = []
    last_start = -1
    for start in starts:
        adj = start
        while adj > last_start + 1 and is_meta_line(lines[adj - 1]):
            adj -= 1
        if adj <= last_start:
            adj = start
        adjusted.append(adj)
        last_start = adj

    # Ensure the first segment includes any leading metadata.
    if adjusted[0] != 0:
        adjusted = [0] + adjusted

    adjusted = sorted(set(adjusted))
    segments: List[str] = []
    for i, start in enumerate(adjusted):
        end = adjusted[i + 1] if i + 1 < len(adjusted) else None
        segment = "\n".join(lines[start:end]).strip()
        if segment:
            segments.append(segment)
    return segments or [text]


def is_select_query(text: str) -> bool:
    # Strip comments and PREFIX/BASE to find the main query verb.
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"(?im)^(\s*(prefix|base)\b.*)$", "", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return False
    # Match the first query keyword.
    match = re.search(
        r"\b(select|construct|ask|describe|insert|delete|with|load|clear|create|drop|copy|move|add)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    return match.group(1).lower() == "select"


def sha256_hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def extract_queries_from_md(text: str) -> List[str]:
    pattern = re.compile(r"```sparql\\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    return [m.group(1) for m in pattern.finditer(text)]


def extract_queries_from_file(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if suffix in {".rq", ".sparql"}:
        return [
            {"source_type": "repo_file", "query": q}
            for q in split_queries(text)
        ]
    if suffix == ".md":
        queries: List[Dict[str, str]] = []
        for block in extract_queries_from_md(text):
            for q in split_queries(block):
                queries.append({"source_type": "md_fence", "query": q})
        return queries
    return []


def resolve_repo_url(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    return repo_url if parsed.scheme else f"https://{repo_url}"


def build_query_record(
    kg_id: str,
    query_type: str,
    raw_query: str,
    clean_query: str,
    raw_hash: str,
    clean_hash: str,
) -> Dict[str, object]:
    return {
        "query_id": f"{kg_id}__{clean_hash}",
        "kg_id": kg_id,
        "query_type": query_type,
        "sparql_raw": raw_query,
        "sparql_clean": clean_query,
        "sparql_hash": clean_hash,
        "raw_hash": raw_hash,
        "evidence": [],
        "cq_items": [],
        "nl_question": {
            "text": None,
            "source": None,
            "generated_at": None,
            "generator": None,
        },
        "justification": None,
        "comments": None,
        "verification": {"status": "unverified", "notes": None},
        "latest_run": None,
        "latest_successful_run": None,
        "run_history": [],
    }


def main() -> None:
    seeds_path = Path("seeds.yaml")
    out_path = Path("kg_queries.jsonl")
    repos_dir = Path("repos")
    repos_dir.mkdir(parents=True, exist_ok=True)

    raw_kgs = load_seeds(seeds_path)
    kgs = [parse_kg_seed(r) for r in raw_kgs]

    records: List[Dict[str, object]] = []
    record_by_key: Dict[tuple[str, str], Dict[str, object]] = {}
    label_counters: Dict[str, int] = {}
    extracted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for kg in kgs:
        for repo_url in kg.repos:
            repo_url = resolve_repo_url(repo_url)
            repo_dir = ensure_repo_cloned(repo_url, repos_dir)
            repo_commit = get_repo_commit(repo_dir)

            for path in iter_repo_files(repo_dir):
                extracted = extract_queries_from_file(path)
                if not extracted:
                    continue
                rel_path = str(path.relative_to(repo_dir))
                for item in extracted:
                    normalized = normalize_query(item["query"])
                    if not normalized:
                        continue
                    if not is_select_query(normalized):
                        continue
                    clean_hash = sha256_hash(normalized)
                    raw_hash = sha256_hash(item["query"])
                    key = (kg.kg_id, clean_hash)
                    if key not in record_by_key:
                        label_counters[kg.kg_id] = label_counters.get(kg.kg_id, 0) + 1
                        query_label = f"{kg.kg_id}-{label_counters[kg.kg_id]:04d}"
                        record_by_key[key] = build_query_record(
                            kg_id=kg.kg_id,
                            query_type="select",
                            raw_query=item["query"],
                            clean_query=normalized,
                            raw_hash=raw_hash,
                            clean_hash=clean_hash,
                        )
                        record_by_key[key]["query_label"] = query_label
                        records.append(record_by_key[key])
                    record = record_by_key[key]
                    record["evidence"].append(
                        {
                            "evidence_id": f"e{len(record['evidence']) + 1}",
                            "type": item["source_type"],
                            "source_url": repo_url,
                            "source_path": rel_path,
                            "repo_commit": repo_commit,
                            "snippet": item["query"].strip(),
                            "extracted_at": extracted_at,
                            "extractor_version": "extract_queries.py@v1",
                        }
                    )

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path.resolve()}")


if __name__ == "__main__":
    main()
