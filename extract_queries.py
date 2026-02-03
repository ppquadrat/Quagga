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
import html
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import yaml
from pypdf import PdfReader


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


def load_kgs_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


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
    # Capture fenced blocks with or without language tag; SPARQL will be filtered later.
    pattern = re.compile(r"```(?:sparql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    return [m.group(1) for m in pattern.finditer(text)]


def extract_queries_from_pre(text: str) -> List[str]:
    pattern = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
    return [html.unescape(m.group(1)) for m in pattern.finditer(text)]


def extract_queries_from_py(text: str) -> List[str]:
    pattern = re.compile(r"(['\"]{3})(.*?)\1", re.DOTALL)
    return [m.group(2) for m in pattern.finditer(text)]


def extract_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def extract_queries_from_pdf_text(text: str) -> List[str]:
    lines = text.splitlines()
    blocks: List[str] = []
    current: List[str] = []
    in_block = False
    depth = 0
    seen_query = False
    for line in lines:
        stripped = line.strip()
        is_code_line = bool(
            re.search(r"\b(select|construct|ask|describe|where|prefix)\b", stripped, re.IGNORECASE)
            or "{ " in stripped
            or stripped.startswith("{")
            or stripped.startswith("PREFIX")
        )
        if is_code_line:
            if not in_block:
                in_block = True
                current = []
                depth = 0
                seen_query = False
            current.append(line.rstrip())
            if re.search(r"\b(select|construct|ask|describe)\b", stripped, re.IGNORECASE):
                seen_query = True
            depth += line.count("{") - line.count("}")
            continue
        if in_block:
            # Keep collecting until braces close after a query start.
            depth += line.count("{") - line.count("}")
            current.append(line.rstrip())
            if seen_query and depth <= 0:
                if current:
                    blocks.append("\n".join(current).strip())
                current = []
                in_block = False
                depth = 0
                seen_query = False
                continue
    if in_block and current:
        blocks.append("\n".join(current).strip())
    # Repair broken PREFIX lines split across PDF line breaks.
    fixed_blocks: List[str] = []
    for block in blocks:
        lines = block.splitlines()
        merged: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if (
                line.strip().lower().startswith("prefix")
                and i + 1 < len(lines)
                and lines[i + 1].strip().startswith("<")
            ):
                line = f"{line} {lines[i + 1].strip()}"
                i += 2
                merged.append(line)
                continue
            if line.strip().endswith(":") and i + 1 < len(lines) and lines[i + 1].strip().startswith("<"):
                line = f"{line} {lines[i + 1].strip()}"
                i += 2
                merged.append(line)
                continue
            if "<" in line and ">" not in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not re.match(r"^\s*(prefix|select|construct|ask|describe)\b", next_line, re.IGNORECASE):
                    line = f"{line}{next_line}"
                    i += 2
                    merged.append(line)
                    continue
            merged.append(line)
            i += 1
        fixed_blocks.append("\n".join(merged).strip())
    queries: List[str] = []
    for block in fixed_blocks:
        for q in split_queries(block):
            queries.append(q)
    return queries


def parse_source_file(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.startswith("SOURCE:"):
        parts = text.split("\n\n", 1)
        header = parts[0].strip()
        body = parts[1] if len(parts) > 1 else ""
        url = header.replace("SOURCE:", "").strip()
        return {"url": url, "text": normalize_source_text(body), "raw": body}
    return {"url": "", "text": normalize_source_text(text), "raw": text}


def normalize_source_text(text: str) -> str:
    if "<html" in text.lower() or "markdown-body" in text.lower():
        return html_to_markdownish(text)
    return text


def html_to_markdownish(text: str) -> str:
    def extract_markdown_div(html_text: str) -> str:
        start_match = re.search(
            r'<div[^>]*class="[^"]*markdown-body[^"]*"[^>]*>',
            html_text,
            re.IGNORECASE,
        )
        if not start_match:
            return ""
        start_idx = start_match.end()
        depth = 1
        idx = start_idx
        div_open = re.compile(r"<div[^>]*>", re.IGNORECASE)
        div_close = re.compile(r"</div>", re.IGNORECASE)
        while idx < len(html_text):
            next_open = div_open.search(html_text, idx)
            next_close = div_close.search(html_text, idx)
            if not next_close:
                break
            if next_open and next_open.start() < next_close.start():
                depth += 1
                idx = next_open.end()
            else:
                depth -= 1
                idx = next_close.end()
                if depth == 0:
                    return html_text[start_idx:next_close.start()]
        return ""

    match = re.search(r'<article class="markdown-body[^"]*">(.*?)</article>', text, re.DOTALL | re.IGNORECASE)
    if match:
        body = match.group(1)
    else:
        match = re.search(r'<article[^>]*itemprop="text"[^>]*>(.*?)</article>', text, re.DOTALL | re.IGNORECASE)
        if match:
            body = match.group(1)
        else:
            md_div = extract_markdown_div(text)
            if md_div:
                body = md_div
            else:
                match = re.search(r'<div[^>]*id="readme"[^>]*>(.*?)</div>', text, re.DOTALL | re.IGNORECASE)
                body = match.group(1) if match else text

    body = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<pre[^>]*><code[^>]*>", "```\n", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"</code></pre>", "\n```", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<br\\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</p>", "\n\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<p[^>]*>", "", body, flags=re.IGNORECASE)

    for level in range(6, 0, -1):
        pattern = re.compile(rf"<h{level}[^>]*>(.*?)</h{level}>", re.DOTALL | re.IGNORECASE)
        body = pattern.sub(lambda m: "\n" + ("#" * level) + " " + re.sub(r"<[^>]+>", "", m.group(1)).strip() + "\n", body)

    body = re.sub(r"<li[^>]*>(.*?)</li>", lambda m: "- " + re.sub(r"<[^>]+>", "", m.group(1)).strip() + "\n", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<[^>]+>", "", body)
    body = html.unescape(body)
    return body


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
        for block in extract_queries_from_pre(text):
            for q in split_queries(block):
                queries.append({"source_type": "md_pre", "query": q})
        return queries
    if suffix == ".py":
        queries = []
        for block in extract_queries_from_py(text):
            for q in split_queries(block):
                queries.append({"source_type": "repo_file", "query": q})
        return queries
    return []


def resolve_repo_url(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    return repo_url if parsed.scheme else f"https://{repo_url}"


def build_query_record(
    kg_id: str,
    query_label: str,
    query_type: str,
    raw_query: str,
    clean_query: str,
    raw_hash: str,
    clean_hash: str,
) -> Dict[str, object]:
    return {
        "query_label": query_label,
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
    kgs_path = Path("kgs.jsonl")
    pdfs_dir = Path("pdfs")

    raw_kgs = load_seeds(seeds_path)
    kgs = [parse_kg_seed(r) for r in raw_kgs]

    records: List[Dict[str, object]] = []
    record_by_key: Dict[tuple[str, str], Dict[str, object]] = {}
    label_counters: Dict[str, int] = {}
    extracted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    kg_sources: Dict[str, List[str]] = {}
    if kgs_path.exists():
        for kg in load_kgs_jsonl(kgs_path):
            kg_id = kg.get("kg_id")
            source_files = kg.get("source_files")
            docs = kg.get("docs")
            if isinstance(kg_id, str) and isinstance(source_files, list):
                filtered = []
                for s in source_files:
                    if not isinstance(s, str):
                        continue
                    # Skip README snapshots pulled from GitHub API to avoid duplicates.
                    if "api-github-com" in s:
                        continue
                    filtered.append(s)
                kg_sources[kg_id] = filtered
            if isinstance(kg_id, str) and isinstance(docs, list):
                for doc in docs:
                    if not isinstance(doc, str):
                        continue
                    doc_path = Path(doc)
                    if doc_path.exists():
                        kg_sources.setdefault(kg_id, []).append(str(doc_path))

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
                            query_label=query_label,
                            query_type="select",
                            raw_query=item["query"],
                            clean_query=normalized,
                            raw_hash=raw_hash,
                            clean_hash=clean_hash,
                        )
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

        # Extract from local PDFs, keyed by filename match on kg_id or explicit doc paths.
        pdf_paths: List[Path] = []
        if pdfs_dir.exists():
            pdf_paths.extend(pdfs_dir.glob("*.pdf"))
        for doc_path in kg_sources.get(kg.kg_id, []):
            path_obj = Path(doc_path)
            if path_obj.suffix.lower() == ".pdf" and path_obj.exists():
                pdf_paths.append(path_obj)
        seen_pdfs: set[Path] = set()
        for pdf_path in pdf_paths:
            if pdf_path in seen_pdfs:
                continue
            seen_pdfs.add(pdf_path)
            if kg.kg_id.lower() not in pdf_path.name.lower() and str(pdf_path) not in kg_sources.get(kg.kg_id, []):
                continue
            pdf_text = extract_text_from_pdf(pdf_path)
            if not pdf_text.strip():
                continue
            for q in extract_queries_from_pdf_text(pdf_text):
                normalized = normalize_query(q)
                if not normalized:
                    continue
                if not is_select_query(normalized):
                    continue
                clean_hash = sha256_hash(normalized)
                raw_hash = sha256_hash(q)
                key = (kg.kg_id, clean_hash)
                if key not in record_by_key:
                    label_counters[kg.kg_id] = label_counters.get(kg.kg_id, 0) + 1
                    query_label = f"{kg.kg_id}-{label_counters[kg.kg_id]:04d}"
                    record_by_key[key] = build_query_record(
                        kg_id=kg.kg_id,
                        query_label=query_label,
                        query_type="select",
                        raw_query=q,
                        clean_query=normalized,
                        raw_hash=raw_hash,
                        clean_hash=clean_hash,
                    )
                    records.append(record_by_key[key])
                record = record_by_key[key]
                record["evidence"].append(
                    {
                        "evidence_id": f"e{len(record['evidence']) + 1}",
                        "type": "doc_pdf",
                        "source_url": "",
                        "source_path": str(pdf_path),
                        "repo_commit": "",
                        "snippet": q.strip(),
                        "extracted_at": extracted_at,
                        "extractor_version": "extract_queries.py@v1",
                    }
                )

        for src_file in kg_sources.get(kg.kg_id, []):
            src_path = Path(src_file)
            if not src_path.is_absolute():
                if src_path.parts and src_path.parts[0] == "kg_sources":
                    src_path = src_path
                else:
                    src_path = Path("kg_sources") / src_path
            if not src_path.exists():
                continue
            parsed = parse_source_file(src_path)
            body = parsed["text"]
            raw_body = parsed["raw"]
            source_url = parsed["url"]
            if not body.strip():
                continue
            blocks = extract_queries_from_md(body)
            if "<pre" in raw_body.lower():
                blocks += extract_queries_from_pre(raw_body)
            for block in blocks:
                for q in split_queries(block):
                    normalized = normalize_query(q)
                    if not normalized:
                        continue
                    if not is_select_query(normalized):
                        continue
                    clean_hash = sha256_hash(normalized)
                    raw_hash = sha256_hash(q)
                    key = (kg.kg_id, clean_hash)
                    if key not in record_by_key:
                        label_counters[kg.kg_id] = label_counters.get(kg.kg_id, 0) + 1
                        query_label = f"{kg.kg_id}-{label_counters[kg.kg_id]:04d}"
                        record_by_key[key] = build_query_record(
                            kg_id=kg.kg_id,
                            query_label=query_label,
                            query_type="select",
                            raw_query=q,
                            clean_query=normalized,
                            raw_hash=raw_hash,
                            clean_hash=clean_hash,
                        )
                        records.append(record_by_key[key])
                    record = record_by_key[key]
                    record["evidence"].append(
                        {
                            "evidence_id": f"e{len(record['evidence']) + 1}",
                            "type": "doc_pre" if "<pre" in raw_body.lower() else "doc_fence",
                            "source_url": source_url,
                            "source_path": str(src_path),
                            "repo_commit": "",
                            "snippet": q.strip(),
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
