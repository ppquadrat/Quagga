#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
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
    pattern = re.compile(r"```sparql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    return [m.group(1) for m in pattern.finditer(text)]


def extract_queries_from_file(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if suffix in {".rq", ".sparql"}:
        return [{"source_type": "repo_file", "query": text}]
    if suffix == ".md":
        return [{"source_type": "md_fence", "query": q} for q in extract_queries_from_md(text)]
    return []


def resolve_repo_url(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    return repo_url if parsed.scheme else f"https://{repo_url}"


def main() -> None:
    seeds_path = Path("seeds.yaml")
    out_path = Path("raw_queries.jsonl")
    repos_dir = Path("repos")
    repos_dir.mkdir(parents=True, exist_ok=True)

    raw_kgs = load_seeds(seeds_path)
    kgs = [parse_kg_seed(r) for r in raw_kgs]

    records: List[Dict[str, str]] = []
    seen_hashes: set[str] = set()

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
                    digest = sha256_hash(normalized)
                    if digest in seen_hashes:
                        continue
                    seen_hashes.add(digest)
                    records.append(
                        {
                            "kg_id": kg.kg_id,
                            "source_type": item["source_type"],
                            "repo_url": repo_url,
                            "repo_commit": repo_commit,
                            "path": rel_path,
                            "query": normalized,
                            "hash": digest,
                        }
                    )

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path.resolve()}")


if __name__ == "__main__":
    main()
