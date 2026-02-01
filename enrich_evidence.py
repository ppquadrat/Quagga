#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
import html
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


def load_query_records(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("kg_queries.jsonl must be a JSON array or JSONL.")
        return data
    records: List[Dict[str, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def repo_dir_from_url(repo_url: str) -> Path:
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Bad repo URL: {repo_url}")
    owner, repo = parts[-2], parts[-1]
    return Path(f"{owner}__{repo}")


def resolve_repo_url(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    return repo_url if parsed.scheme else f"https://{repo_url}"


def iter_repo_files(repo_dir: Path) -> Iterable[Path]:
    for path in repo_dir.rglob("*"):
        if path.is_file():
            yield path


def split_queries_with_starts(text: str) -> List[Dict[str, object]]:
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
        return [{"start": 0, "query": text}]

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

    if adjusted[0] != 0:
        adjusted = [0] + adjusted

    adjusted = sorted(set(adjusted))
    segments: List[Dict[str, object]] = []
    for i, start in enumerate(adjusted):
        end = adjusted[i + 1] if i + 1 < len(adjusted) else None
        segment = "\n".join(lines[start:end]).strip()
        if segment:
            segments.append({"start": start, "query": segment})
    return segments or [{"start": 0, "query": text}]


def sha256_hash(text: str) -> str:
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def normalize_query(text: str) -> str:
    normalized = text.strip()
    while normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    return normalized


def clean_md_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        if stripped.startswith(("-", "*")):
            stripped = stripped[1:].strip()
        stripped = re.sub(r"^\d+\.", "", stripped).strip()
        lines.append(stripped)
    return " ".join(lines).strip()


def extract_recent_text_blocks(prefix: str, limit: int = 2) -> str:
    if "<" in prefix:
        prefix = re.sub(r"<pre[^>]*>.*?</pre>", "", prefix, flags=re.DOTALL | re.IGNORECASE)
    normalized = html_to_markdownish(prefix) if "<" in prefix else prefix
    normalized = re.sub(r"```.*?```", "", normalized, flags=re.DOTALL)
    parts = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    if not parts:
        return ""
    return "\n".join(parts[-limit:]).strip()


def extract_md_blocks_with_desc(text: str) -> List[Dict[str, object]]:
    pattern = re.compile(r"```sparql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    results: List[Dict[str, str]] = []
    matches = list(pattern.finditer(text))
    for match in matches:
        block = match.group(1)
        prefix = text[: match.start()]
        desc = extract_recent_text_blocks(prefix, limit=1)
        results.append({"query": block, "desc": desc, "start_idx": match.start()})
    return results


def extract_pre_blocks_with_desc(text: str) -> List[Dict[str, object]]:
    pattern = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
    results: List[Dict[str, str]] = []
    matches = list(pattern.finditer(text))
    for match in matches:
        block = html.unescape(match.group(1))
        prefix = text[: match.start()]
        bullet = extract_last_bullet(prefix)
        desc = bullet or extract_recent_text_blocks(prefix, limit=2)
        results.append({"query": block, "desc": desc, "start_idx": match.start()})
    return results


def extract_preceding_comments(lines: List[str], start_idx: int) -> str:
    comments: List[str] = []
    idx = start_idx - 1
    while idx >= 0:
        line = lines[idx].strip()
        if not line:
            idx -= 1
            continue
        if line.startswith("#") or line.startswith("//"):
            comments.append(line.lstrip("#/").strip())
            idx -= 1
            continue
        if line.endswith("*/") or line.startswith("/*"):
            block_lines: List[str] = []
            while idx >= 0:
                block_line = lines[idx].strip()
                cleaned = block_line.lstrip("/*").rstrip("*/").strip()
                if cleaned:
                    block_lines.append(cleaned)
                if block_line.startswith("/*"):
                    break
                idx -= 1
            comments.extend(reversed(block_lines))
            idx -= 1
            continue
        break
    comments.reverse()
    return " ".join([c for c in comments if c]).strip()


def extract_leading_context(segment_text: str) -> str:
    lines = segment_text.splitlines()
    keyword_re = re.compile(
        r"^\s*(select|construct|ask|describe|insert|delete|with|load|clear|create|drop|copy|move|add)\b",
        re.IGNORECASE,
    )
    context_lines: List[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if keyword_re.match(line):
            break
        if in_block:
            context_lines.append(line.rstrip())
            if "*/" in stripped:
                in_block = False
            continue
        if stripped.startswith("/*"):
            in_block = True
            context_lines.append(line.rstrip())
            if "*/" in stripped:
                in_block = False
            continue
        if stripped.startswith("#") or stripped.startswith("//"):
            context_lines.append(line.rstrip())
            continue
        if stripped.lower().startswith(("prefix ", "base ")):
            break
        if not stripped:
            context_lines.append("")
            continue
        # Stop at first non-comment, non-prefix content.
        break
    return "\n".join(context_lines).strip()


def add_evidence(
    record: Dict[str, object],
    evidence_type: str,
    source_url: str,
    source_path: str,
    repo_commit: str,
    snippet: str,
    extracted_at: str,
) -> None:
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    snippet = snippet.strip()
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        if (
            ev.get("type") == evidence_type
            and ev.get("source_path") == source_path
            and ev.get("snippet") == snippet
        ):
            record["evidence"] = evidence
            return
    evidence_id = f"e{len(evidence) + 1}"
    evidence.append(
        {
            "evidence_id": evidence_id,
            "type": evidence_type,
            "source_url": source_url,
            "source_path": source_path,
            "repo_commit": repo_commit,
            "snippet": snippet,
            "extracted_at": extracted_at,
            "extractor_version": "enrich_evidence.py@v1",
        }
    )
    record["evidence"] = evidence


def parse_source_file(path: Path) -> Tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.startswith("SOURCE:"):
        parts = text.split("\n\n", 1)
        header = parts[0].strip()
        body = parts[1] if len(parts) > 1 else ""
        url = header.replace("SOURCE:", "").strip()
        return url, normalize_source_text(body), body
    return "", normalize_source_text(text), text


def normalize_source_text(text: str) -> str:
    if "<html" in text.lower() or "markdown-body" in text.lower():
        return html_to_markdownish(text)
    return text


def html_to_markdownish(text: str) -> str:
    def extract_markdown_div(html_text: str) -> Optional[str]:
        start_match = re.search(
            r'<div[^>]*class="[^"]*markdown-body[^"]*"[^>]*>',
            html_text,
            re.IGNORECASE,
        )
        if not start_match:
            return None
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
        return None

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

    # Drop scripts/styles to reduce noise.
    body = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL | re.IGNORECASE)

    def strip_tags(s: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", s))

    # Convert tables to markdown-like rows.
    def convert_tables(s: str) -> str:
        def row_to_md(row_html: str) -> str:
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.DOTALL | re.IGNORECASE)
            if not cells:
                return ""
            cell_text = [strip_tags(c).strip() for c in cells]
            return "| " + " | ".join(cell_text) + " |"

        def table_repl(match_obj: re.Match) -> str:
            table_html = match_obj.group(1)
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE)
            md_rows = [row_to_md(r) for r in rows]
            md_rows = [r for r in md_rows if r]
            return "\n".join(md_rows) + "\n\n" if md_rows else ""

        return re.sub(r"<table[^>]*>(.*?)</table>", table_repl, s, flags=re.DOTALL | re.IGNORECASE)

    body = convert_tables(body)
    body = re.sub(r"<pre[^>]*><code[^>]*>", "```\n", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"</code></pre>", "\n```", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<br\\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</p>", "\n\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<p[^>]*>", "", body, flags=re.IGNORECASE)

    for level in range(6, 0, -1):
        pattern = re.compile(rf"<h{level}[^>]*>(.*?)</h{level}>", re.DOTALL | re.IGNORECASE)
        body = pattern.sub(lambda m: "\n" + ("#" * level) + " " + strip_tags(m.group(1)).strip() + "\n", body)

    body = re.sub(r"<li[^>]*>(.*?)</li>", lambda m: "- " + strip_tags(m.group(1)).strip() + "\n", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<[^>]+>", "", body)
    body = html.unescape(body)
    return body


def extract_cq_section(text: str) -> Optional[str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "competency question" in line.lower() or line.strip().lower().startswith("cq"):
            # capture a small subsection
            snippet_lines: List[str] = []
            for j in range(idx, min(idx + 30, len(lines))):
                snippet_lines.append(lines[j])
                if lines[j].strip().startswith("#") and j > idx:
                    break
            snippet = clean_md_text("\n".join(snippet_lines))
            return snippet if snippet else None
    return None


def extract_heading_bullets(text: str) -> List[str]:
    lines = text.splitlines()
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    keywords = ("competency question", "competency questions", "cqs", "questions")
    results: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        lower = line.strip().lower()
        if heading_re.match(line) and any(k in lower for k in keywords):
            i += 1
            block: List[str] = []
            while i < len(lines):
                if heading_re.match(lines[i]):
                    break
                if lines[i].strip().startswith(("-", "*")):
                    block.append(lines[i].rstrip())
                i += 1
            if block:
                results.append("\n".join(block).strip())
        else:
            i += 1
    return results


def extract_label_blocks(text: str) -> List[str]:
    lines = text.splitlines()
    label_re = re.compile(r"^\s*[A-Z]{2,3}\d+\.\s+")
    blocks: List[str] = []
    i = 0
    while i < len(lines):
        if label_re.match(lines[i]):
            start = i
            i += 1
            while i < len(lines):
                if label_re.match(lines[i]) or lines[i].strip().startswith("```") or lines[i].strip().startswith("<pre>"):
                    break
                if lines[i].strip().startswith("#"):
                    break
                i += 1
            block = "\n".join([ln.rstrip() for ln in lines[start:i] if ln.strip()]).strip()
            if block:
                blocks.append(block)
        else:
            i += 1
    return blocks


def extract_table_blocks(text: str) -> List[str]:
    lines = text.splitlines()
    blocks: List[str] = []
    i = 0
    while i < len(lines) - 1:
        if "|" in lines[i] and "|" in lines[i + 1]:
            header = [c.strip().lower() for c in lines[i].strip("|").split("|")]
            if any("question" in h or "cq" in h or "competency" in h for h in header):
                i += 2
                rows: List[str] = []
                while i < len(lines) and "|" in lines[i]:
                    row = [c.strip() for c in lines[i].strip("|").split("|")]
                    if any(row):
                        rows.append(" | ".join([c for c in row if c]))
                    i += 1
                if rows:
                    blocks.append("\n".join(rows).strip())
                continue
        i += 1
    return blocks


def extract_cq_block(text: str) -> Optional[str]:
    pattern = re.compile(
        r"(#{1,6}\s+.*competency question.*?)(?=\n\s*#{1,6}\s+|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    block = match.group(1)
    lines = block.splitlines()[1:]
    cleaned = "\n".join([ln.rstrip() for ln in lines if ln.strip()]).strip()
    return cleaned if cleaned else None


def extract_context_for_code(text: str, start_idx: int) -> Optional[str]:
    lines = text.splitlines()
    if start_idx > 0:
        start_idx = len(text[:start_idx].splitlines()) - 1
    if start_idx >= len(lines):
        start_idx = len(lines) - 1
    if start_idx < 0:
        return None
    label_re = re.compile(r"^\s*[A-Z]{2,3}\d+\.\s+")
    # Find nearest label above.
    label_idx = None
    i = start_idx - 1
    while i >= 0:
        if label_re.match(lines[i]):
            label_idx = i
            break
        if lines[i].strip().startswith("#"):
            break
        i -= 1
    if label_idx is not None:
        block = "\n".join([ln.rstrip() for ln in lines[label_idx:start_idx] if ln.strip()]).strip()
        return block if block else None
    # Fallback: grab up to 2 preceding non-empty paragraphs/bullets.
    collected: List[str] = []
    i = start_idx - 1
    while i >= 0 and len(collected) < 2:
        if not lines[i].strip():
            i -= 1
            continue
        if lines[i].strip().startswith(("-", "*")):
            collected.append(lines[i].strip())
            i -= 1
            continue
        # paragraph line
        collected.append(lines[i].strip())
        i -= 1
    if collected:
        collected.reverse()
        return "\n".join(collected).strip()
    return None


def clean_desc(text: str) -> str:
    lines = []
    seen = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith(("PREFIX ", "SELECT ", "CONSTRUCT ", "ASK ", "DESCRIBE ", "WHERE ", "FILTER ")):
            continue
        if stripped in {"{", "}", "};", ";"}:
            continue
        if "<" in stripped or ">" in stripped:
            continue
        if stripped.startswith("{") or stripped.endswith("}"):
            continue
        if stripped in seen:
            continue
        seen.add(stripped)
        lines.append(stripped)
    return "\n".join(lines).strip()


def extract_last_bullet(prefix: str) -> str:
    normalized = html_to_markdownish(prefix) if "<" in prefix else prefix
    normalized = re.sub(r"```.*?```", "", normalized, flags=re.DOTALL)
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.startswith("- "):
            return line[2:].strip()
    return ""


def query_has_repo_evidence(rec: Dict[str, object]) -> bool:
    for ev in rec.get("evidence", []) or []:
        if isinstance(ev, dict) and ev.get("type") in {"repo_file", "md_fence", "md_pre"}:
            return True
    return False


def query_has_doc_evidence(rec: Dict[str, object]) -> bool:
    for ev in rec.get("evidence", []) or []:
        if isinstance(ev, dict) and ev.get("type") in {"doc_pre", "doc_fence"}:
            return True
    return False


def select_llm_context(evidence: List[Dict[str, object]], origin: str) -> List[Dict[str, object]]:
    def pick_by_types(types: List[str], limit: int) -> List[Dict[str, object]]:
        picked: List[Dict[str, object]] = []
        for ev in evidence:
            if ev.get("type") in types and ev.get("snippet"):
                picked.append(ev)
                if len(picked) >= limit:
                    break
        return picked

    context: List[Dict[str, object]] = []
    if origin == "repo":
        context.extend(pick_by_types(["query_comment"], 5))
        context.extend(pick_by_types(["readme_query_desc"], 2))
        context.extend(pick_by_types(["doc_query_desc", "web_query_desc"], 2))
        context.extend(pick_by_types(["doc_cq_section", "cq_list"], 1))
    else:
        context.extend(pick_by_types(["web_query_desc", "doc_query_desc"], 3))
        context.extend(pick_by_types(["doc_cq_section", "cq_list"], 1))
    return context[:6]


def infer_query_origin(evidence: List[Dict[str, object]]) -> str:
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        if ev.get("type") in {"repo_file", "md_fence", "md_pre"}:
            return "repo"
        if ev.get("type") in {"doc_pre", "doc_fence"}:
            return "doc"
    return "unknown"


def dedupe_evidence(evidence: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    deduped: List[Dict[str, object]] = []
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        key = (ev.get("type"), ev.get("source_path"), ev.get("snippet"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)
    return deduped


def main() -> None:
    queries_path = Path("kg_queries.jsonl")
    repos_dir = Path("repos")
    sources_dir = Path("kg_sources")
    kgs_path = Path("kgs.jsonl")

    records = load_query_records(queries_path)
    by_kg_hash: Dict[tuple[str, str], Dict[str, object]] = {}
    for rec in records:
        kg_id = rec.get("kg_id")
        sparql_hash = rec.get("sparql_hash")
        if isinstance(kg_id, str) and isinstance(sparql_hash, str):
            by_kg_hash[(kg_id, sparql_hash)] = rec

    extracted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    kg_repos: Dict[str, List[str]] = {}
    before_counts: Dict[str, int] = {}
    for rec in records:
        kg_id = rec.get("kg_id")
        if not isinstance(kg_id, str):
            continue
        evidence = rec.get("evidence")
        if not isinstance(evidence, list):
            continue
        before_counts[kg_id] = before_counts.get(kg_id, 0) + len(evidence)

    for rec in records:
        kg_id = rec.get("kg_id")
        if not isinstance(kg_id, str):
            continue
        evidence = rec.get("evidence")
        if not isinstance(evidence, list):
            continue
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            repo_url = ev.get("source_url")
            if ev.get("type") in {"repo_file", "md_fence", "md_pre"} and isinstance(repo_url, str):
                kg_repos.setdefault(kg_id, [])
                if repo_url not in kg_repos[kg_id]:
                    kg_repos[kg_id].append(repo_url)

    # Map KG -> source files from kgs.jsonl (if present).
    kg_sources: Dict[str, List[str]] = {}
    if kgs_path.exists():
        for kg in load_query_records(kgs_path):
            kg_id = kg.get("kg_id")
            source_files = kg.get("source_files")
            if isinstance(kg_id, str) and isinstance(source_files, list):
                kg_sources[kg_id] = [s for s in source_files if isinstance(s, str)]

    for rec in records:
        kg_id = rec.get("kg_id")
        if not isinstance(kg_id, str):
            continue
        evidence = rec.get("evidence")
        if not isinstance(evidence, list):
            continue
        repo_evidence = [
            e for e in evidence
            if isinstance(e, dict)
            and e.get("type") in {"repo_file", "md_fence"}
            and isinstance(e.get("source_url"), str)
        ]
        for ev in repo_evidence:
            repo_url = ev.get("source_url")
            source_path = ev.get("source_path")
            repo_commit = ev.get("repo_commit")
            if not isinstance(repo_url, str) or not isinstance(source_path, str):
                continue
            repo_url = resolve_repo_url(repo_url)
            repo_dir = repos_dir / repo_dir_from_url(repo_url)
            file_path = repo_dir / source_path
            if not file_path.exists():
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            if file_path.suffix.lower() in {".rq", ".sparql"}:
                lines = text.splitlines()
                for segment in split_queries_with_starts(text):
                    raw_query = segment["query"]
                    start_idx = int(segment["start"])
                    comment_desc = extract_preceding_comments(lines, start_idx)
                    leading_desc = extract_leading_context(raw_query)
                    if leading_desc:
                        if comment_desc:
                            comment_desc = f"{comment_desc} {leading_desc}".strip()
                        else:
                            comment_desc = leading_desc
                    if not comment_desc:
                        continue
                    normalized = normalize_query(raw_query)
                    if not normalized:
                        continue
                    q_hash = sha256_hash(normalized)
                    target = by_kg_hash.get((kg_id, q_hash))
                    if target is None:
                        continue
                    add_evidence(
                        target,
                        "query_comment",
                        repo_url,
                        source_path,
                        str(repo_commit or ""),
                        comment_desc,
                        extracted_at,
                    )
            elif file_path.suffix.lower() == ".md":
                for block in extract_md_blocks_with_desc(text):
                    for segment in split_queries_with_starts(block["query"]):
                        raw_query = segment["query"]
                        normalized = normalize_query(raw_query)
                        if not normalized:
                            continue
                        q_hash = sha256_hash(normalized)
                        target = by_kg_hash.get((kg_id, q_hash))
                        if target is None:
                            continue
                        desc = block.get("desc", "")
                        context = extract_context_for_code(text, int(block.get("start_idx", 0)))
                        if context:
                            desc = f"{context}\n{desc}".strip() if desc else context
                        if desc:
                            add_evidence(
                                target,
                                "doc_query_desc",
                                repo_url,
                                source_path,
                                str(repo_commit or ""),
                                clean_desc(desc),
                                extracted_at,
                            )
                if file_path.name.lower().startswith("readme"):
                    cq_blocks = []
                    heading_blocks = extract_heading_bullets(text)
                    if heading_blocks:
                        cq_blocks.extend(heading_blocks)
                    else:
                        cq_blocks.extend(extract_label_blocks(text))
                    table_blocks = extract_table_blocks(text)
                    if not cq_blocks and not table_blocks:
                        cq_list = extract_cq_block(text)
                        if cq_list:
                            cq_blocks.append(cq_list)
                    for rec2 in records:
                        if rec2.get("kg_id") != kg_id:
                            continue
                        if query_has_doc_evidence(rec2):
                            continue
                        for block in cq_blocks:
                            add_evidence(
                                rec2,
                                "cq_list",
                                repo_url,
                                source_path,
                                str(repo_commit or ""),
                                block,
                                extracted_at,
                            )
                        for tbl in table_blocks:
                            add_evidence(
                                rec2,
                                "doc_cq_section",
                                repo_url,
                                source_path,
                                str(repo_commit or ""),
                                tbl,
                                extracted_at,
                            )

        # Parse README files explicitly for query descriptions.
        for repo_url in kg_repos.get(kg_id, []):
            repo_url = resolve_repo_url(repo_url)
            repo_dir = repos_dir / repo_dir_from_url(repo_url)
            if not repo_dir.exists():
                continue
            readmes = [p for p in repo_dir.iterdir() if p.is_file() and p.name.lower().startswith("readme")]
            for readme in readmes:
                try:
                    readme_text = readme.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                cq_blocks = []
                heading_blocks = extract_heading_bullets(readme_text)
                if heading_blocks:
                    cq_blocks.extend(heading_blocks)
                else:
                    cq_blocks.extend(extract_label_blocks(readme_text))
                table_blocks = extract_table_blocks(readme_text)
                if not cq_blocks and not table_blocks:
                    cq_list = extract_cq_block(readme_text)
                    if cq_list:
                        cq_blocks.append(cq_list)
                for rec2 in records:
                    if rec2.get("kg_id") != kg_id:
                        continue
                    if query_has_doc_evidence(rec2):
                        continue
                    for block in cq_blocks:
                        add_evidence(
                            rec2,
                            "cq_list",
                            repo_url,
                            str(readme.relative_to(repo_dir)),
                            "",
                            block,
                            extracted_at,
                        )
                    for tbl in table_blocks:
                        add_evidence(
                            rec2,
                            "doc_cq_section",
                            repo_url,
                            str(readme.relative_to(repo_dir)),
                            "",
                            tbl,
                            extracted_at,
                        )
                for block in extract_md_blocks_with_desc(readme_text) + extract_pre_blocks_with_desc(readme_text):
                    for segment in split_queries_with_starts(block["query"]):
                        raw_query = segment["query"]
                        normalized = normalize_query(raw_query)
                        if not normalized:
                            continue
                        q_hash = sha256_hash(normalized)
                        target = by_kg_hash.get((kg_id, q_hash))
                        if target is None:
                            continue
                        if query_has_doc_evidence(target):
                            continue
                        desc = block.get("desc", "")
                        context = extract_context_for_code(readme_text, int(block.get("start_idx", 0)))
                        if context:
                            desc = f"{context}\n{desc}".strip() if desc else context
                        if desc:
                            add_evidence(
                                target,
                                "readme_query_desc",
                                repo_url,
                                str(readme.relative_to(repo_dir)),
                                "",
                                clean_desc(desc),
                                extracted_at,
                            )

        # Enrich from kg_sources (web/papers) if available.
        source_files = kg_sources.get(kg_id, [])
        doc_cq_seen = False
        for src_file in source_files:
            src_path = Path(src_file)
            if not src_path.is_absolute():
                if src_path.parts and src_path.parts[0] == "kg_sources":
                    src_path = src_path
                else:
                    src_path = sources_dir / src_path
            if not src_path.exists():
                continue
            if "api-github-com" in str(src_path):
                continue
            source_url, body, raw_body = parse_source_file(src_path)
            if not body.strip():
                continue
            cq_blocks = []
            heading_blocks = extract_heading_bullets(body)
            if heading_blocks:
                cq_blocks.extend(heading_blocks)
            else:
                cq_blocks.extend(extract_label_blocks(body))
            table_blocks = extract_table_blocks(body)
            if not cq_blocks and not table_blocks:
                cq_list = extract_cq_block(body)
                if cq_list:
                    cq_blocks.append(cq_list)
            for rec2 in records:
                if rec2.get("kg_id") != kg_id:
                    continue
                has_same_source = any(
                    isinstance(e, dict) and e.get("source_path") == str(src_path)
                    for e in rec2.get("evidence", []) or []
                )
                if not (has_same_source or query_has_repo_evidence(rec2)):
                    continue
                for block in cq_blocks:
                    add_evidence(
                        rec2,
                        "cq_list",
                        source_url or "",
                        str(src_path),
                        "",
                        clean_desc(block),
                        extracted_at,
                    )
                for tbl in table_blocks:
                    add_evidence(
                        rec2,
                        "doc_cq_section",
                        source_url or "",
                        str(src_path),
                        "",
                        tbl,
                        extracted_at,
                    )
                if table_blocks:
                    doc_cq_seen = True
            # Try to match SPARQL blocks to queries.
            for block in extract_md_blocks_with_desc(body) + extract_pre_blocks_with_desc(raw_body):
                for segment in split_queries_with_starts(block["query"]):
                    raw_query = segment["query"]
                    normalized = normalize_query(raw_query)
                    if not normalized:
                        continue
                    q_hash = sha256_hash(normalized)
                    target = by_kg_hash.get((kg_id, q_hash))
                    if target is None:
                        continue
                    desc = block.get("desc", "")
                    context = extract_context_for_code(body, int(block.get("start_idx", 0)))
                    if context:
                        desc = f"{context}\n{desc}".strip() if desc else context
                    if desc:
                        add_evidence(
                            target,
                            "web_query_desc",
                            source_url or "",
                            str(src_path),
                            "",
                            clean_desc(desc),
                            extracted_at,
                        )

            cq_section = extract_cq_section(body)
            if cq_section and not doc_cq_seen and not table_blocks:
                for rec2 in records:
                    if rec2.get("kg_id") != kg_id:
                        continue
                    if not any(
                        isinstance(e, dict) and e.get("source_path") == str(src_path)
                        for e in rec2.get("evidence", []) or []
                    ):
                        continue
                    add_evidence(
                        rec2,
                        "doc_cq_section",
                        source_url or "",
                        str(src_path),
                        "",
                        clean_desc(cq_section),
                        extracted_at,
                    )

    after_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    for rec in records:
        evidence = rec.get("evidence")
        if isinstance(evidence, list):
            rec["evidence"] = dedupe_evidence(evidence)
            for ev in rec["evidence"]:
                if isinstance(ev, dict) and ev.get("type"):
                    type_counts[ev["type"]] = type_counts.get(ev["type"], 0) + 1
            kg_id = rec.get("kg_id")
            if isinstance(kg_id, str):
                after_counts[kg_id] = after_counts.get(kg_id, 0) + len(rec["evidence"])
            origin = infer_query_origin(rec["evidence"])
            rec["llm_context"] = select_llm_context(rec["evidence"], origin)

    write_jsonl(queries_path, records)
    print(f"Wrote {len(records)} records to {queries_path.resolve()}")
    if after_counts:
        print("\nEvidence counts by KG:")
        for kg_id in sorted(after_counts):
            before = before_counts.get(kg_id, 0)
            after = after_counts.get(kg_id, 0)
            delta = after - before
            print(f"- {kg_id}: {after} (delta={delta})")
    if type_counts:
        print("\nEvidence counts by type:")
        for etype in sorted(type_counts):
            print(f"- {etype}: {type_counts[etype]}")


if __name__ == "__main__":
    main()
