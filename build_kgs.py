#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import yaml


@dataclass
class SparqlConfig:
    endpoint: str
    auth: str = "none"
    expected_namespaces: Optional[List[str]] = None


@dataclass
class KGSeed:
    kg_id: str
    name: str
    project: Optional[str] = None
    description_hint: Optional[str] = None
    sparql: Optional[SparqlConfig] = None
    repos: List[str] = None
    docs: List[str] = None
    priority: Optional[str] = None
    notes: Optional[str] = None


def slugify_filename(text: str, max_len: int = 80) -> str:
    """Make a filesystem-friendly slug from a URL or label."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:max_len] if len(text) > max_len else text


def fetch_url_text(url: str, timeout_s: int = 20) -> Dict[str, Any]:
    """Fetch a URL as text. Returns dict with url, text, and optional error."""
    try:
        r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "kg-pipeline/0.1"})
        if r.status_code != 200:
            return {"url": url, "text": "", "error": f"http_{r.status_code}"}
        return {"url": url, "text": r.text}
    except requests.RequestException as e:
        return {"url": url, "text": "", "error": f"request_error:{e.__class__.__name__}"}


def fetch_github_readme(repo_url: str, timeout_s: int = 20) -> Dict[str, Any]:
    """Fetch a GitHub repo README via GitHub API (raw), with raw fallback."""
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        return {"url": repo_url, "text": "", "error": "bad_repo_url"}
    owner, repo = parts[-2], parts[-1]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        headers = {
            "Accept": "application/vnd.github.raw",
            "User-Agent": "kg-pipeline/0.1",
        }
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.get(api_url, timeout=timeout_s, headers=headers)
        if r.status_code != 200:
            api_error = {"url": api_url, "text": "", "error": f"http_{r.status_code}"}
        else:
            return {"url": api_url, "text": r.text}
    except requests.RequestException as e:
        api_error = {"url": api_url, "text": "", "error": f"request_error:{e.__class__.__name__}"}

    # Fallback: try raw GitHub URLs without using the API.
    raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD"
    candidates = [
        "README.md",
        "README.rst",
        "README.txt",
        "README",
        "readme.md",
        "readme.rst",
        "readme.txt",
        "readme",
    ]
    for name in candidates:
        raw_url = f"{raw_base}/{name}"
        try:
            r = requests.get(
                raw_url,
                timeout=timeout_s,
                headers={"User-Agent": "kg-pipeline/0.1"},
            )
            if r.status_code == 200:
                return {"url": raw_url, "text": r.text}
        except requests.RequestException:
            continue

    return api_error


def save_sources(kg_id: str, sources: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    """Save each source text to a file. Returns list of file paths (as strings)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    for idx, src in enumerate(sources, start=1):
        url = src.get("url") or src.get("source_url") or ""
        text = (src.get("text") or "").strip()
        if not text:
            continue

        netloc = urlparse(url).netloc
        base = slugify_filename(netloc) if netloc else "source"
        fname = f"{kg_id}__{idx:02d}__{base}.txt"
        path = out_dir / fname

        path.write_text(f"SOURCE: {url}\n\n{text}", encoding="utf-8")
        saved.append(str(path))

    return saved


def load_seeds(path: Path) -> List[Dict[str, Any]]:
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


def parse_kg_seed(raw: Dict[str, Any]) -> KGSeed:
    kg_id = raw.get("kg_id")
    name = raw.get("name")
    if not isinstance(kg_id, str) or not kg_id.strip():
        raise ValueError("Each KG must have a non-empty string 'kg_id'.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"KG '{kg_id}': must have a non-empty string 'name'.")

    repos = raw.get("repos") or []
    docs = raw.get("docs") or []
    if not isinstance(repos, list) or not all(isinstance(x, str) for x in repos):
        raise ValueError(f"KG '{kg_id}': 'repos' must be a list of strings.")
    if not isinstance(docs, list) or not all(isinstance(x, str) for x in docs):
        raise ValueError(f"KG '{kg_id}': 'docs' must be a list of strings.")

    sparql_cfg = None
    sparql = raw.get("sparql")
    if sparql is not None:
        if not isinstance(sparql, dict):
            raise ValueError(f"KG '{kg_id}': 'sparql' must be a mapping (dict).")
        endpoint = sparql.get("endpoint")
        auth = sparql.get("auth", "none")
        expected_namespaces = sparql.get("expected_namespaces")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError(f"KG '{kg_id}': sparql.endpoint must be a non-empty string.")
        if not isinstance(auth, str):
            raise ValueError(f"KG '{kg_id}': sparql.auth must be a string.")
        if expected_namespaces is not None:
            if not isinstance(expected_namespaces, list) or not all(
                isinstance(x, str) for x in expected_namespaces
            ):
                raise ValueError(
                    f"KG '{kg_id}': sparql.expected_namespaces must be a list of strings."
                )
        sparql_cfg = SparqlConfig(
            endpoint=endpoint.strip(),
            auth=auth.strip(),
            expected_namespaces=expected_namespaces,
        )

    return KGSeed(
        kg_id=kg_id.strip(),
        name=name.strip(),
        project=raw.get("project"),
        description_hint=raw.get("description_hint"),
        sparql=sparql_cfg,
        repos=repos,
        docs=docs,
        priority=raw.get("priority"),
        notes=raw.get("notes"),
    )


def kgseed_to_record(kg: KGSeed) -> Dict[str, Any]:
    today = date.today().isoformat()
    sparql_obj = None
    if kg.sparql:
        sparql_obj = {
            "endpoint": kg.sparql.endpoint,
            "auth": kg.sparql.auth,
            "graph": None,
        }
        if kg.sparql.expected_namespaces:
            sparql_obj["expected_namespaces"] = list(kg.sparql.expected_namespaces)
    return {
        "kg_id": kg.kg_id,
        "name": kg.name,
        "project": kg.project,
        "description": None,
        "sparql": sparql_obj,
        "dataset": {"dump_url": None, "local_path": None, "format": None},
        "repos": list(kg.repos or []),
        "docs": list(kg.docs or []),
        "notes": kg.notes,
        "created_at": today,
        "updated_at": today,
        "description_hint": kg.description_hint,
        "priority": kg.priority,
    }


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def fetch_sources_for_kg(kg: KGSeed) -> List[Dict[str, Any]]:
    """Fetch README + docs sources for a KG (deterministic, no OpenAI)."""
    sources: List[Dict[str, Any]] = []

    for repo_url in (kg.repos or []):
        if "github.com" in repo_url:
            sources.append(fetch_github_readme(repo_url))
        else:
            sources.append(fetch_url_text(repo_url))

    for doc_url in (kg.docs or []):
        normalized = doc_url
        if "github.com/" in doc_url and "/blob/" in doc_url:
            # Convert GitHub blob URLs to raw content.
            parts = doc_url.split("github.com/", 1)[-1]
            normalized = "https://raw.githubusercontent.com/" + parts.replace("/blob/", "/")
        sources.append(fetch_url_text(normalized))

    return sources


def main() -> None:
    seeds_path = Path("seeds.yaml")
    out_path = Path("kgs.jsonl")
    sources_dir = Path("kg_sources")

    raw_kgs = load_seeds(seeds_path)
    kgs = [parse_kg_seed(r) for r in raw_kgs]

    print(f"\nStep 1: loaded {len(kgs)} KGs from {seeds_path.resolve()}\n")

    print("Step 2: fetching README/doc sources (deterministic, no OpenAI yet)\n")
    sources_by_kg: Dict[str, List[Dict[str, Any]]] = {}
    saved_by_kg: Dict[str, List[str]] = {}
    for kg in kgs:
        sources = fetch_sources_for_kg(kg)
        saved_files = save_sources(kg.kg_id, sources, sources_dir)
        sources_by_kg[kg.kg_id] = sources
        saved_by_kg[kg.kg_id] = saved_files

        endpoint = kg.sparql.endpoint if kg.sparql else "(no endpoint)"
        print(f"- {kg.kg_id}: {kg.name}")
        print(f"  endpoint: {endpoint}")
        print(f"  repos:    {len(kg.repos or [])}")
        print(f"  docs:     {len(kg.docs or [])}")
        print(f"  sources:  {len(saved_files)} saved\n")

    print("Step 3: writing kgs.jsonl (metadata + provenance, no generated descriptions yet)\n")
    records: List[Dict[str, Any]] = []
    for kg in kgs:
        rec = kgseed_to_record(kg)
        sources = sources_by_kg.get(kg.kg_id, [])
        saved_files = saved_by_kg.get(kg.kg_id, [])

        rec["source_urls"] = [s.get("url") for s in sources if s.get("url")]
        rec["source_files"] = saved_files
        records.append(rec)

    write_jsonl(out_path, records)
    print(f"Wrote {len(records)} records to {out_path.resolve()}")


if __name__ == "__main__":
    main()
