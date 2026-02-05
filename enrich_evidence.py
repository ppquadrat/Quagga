#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
import html
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from pypdf import PdfReader


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
    pattern = re.compile(r"```(?:sparql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
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
    if not snippet:
        return
    if evidence_type == "cq_item":
        snippet = clean_desc(snippet)
        if not snippet:
            return
        if re.match(r"^\s*(table|figure|algorithm)\s+\d+[:.].*competency\s+questions", snippet, re.IGNORECASE):
            return
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


def extract_pdf_captions(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    captions: List[str] = []
    caption_re = re.compile(r"^(figure|fig\.?|table|tab\.)\s*\d+", re.IGNORECASE)
    for i, line in enumerate(lines):
        if not caption_re.match(line):
            continue
        # Heuristic: caption lines are short and often end with '.' or ':'.
        if len(line.split()) < 3:
            continue
        # Capture this line plus up to 2 following lines unless they look like body text.
        collected = [line]
        for j in range(i + 1, min(i + 4, len(lines))):
            nxt = lines[j]
            if caption_re.match(nxt):
                break
            if len(nxt.split()) > 30:
                break
            collected.append(nxt)
        captions.append(" ".join(collected).strip())
    return captions


def extract_pdf_cq_captions(text: str) -> List[str]:
    # Capture captions even if line breaks split them.
    candidates: List[str] = []
    pattern = re.compile(
        r"(table\s*\d+\s*[:\-]?[\s\S]{0,80}competency\s+questions[^\n]{0,80})",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        snippet = " ".join(match.group(1).split())
        candidates.append(snippet)
    return candidates


def extract_pdf_tables_for_captions(text: str, captions: List[str]) -> List[str]:
    if not text.strip() or not captions:
        return []
    lines = text.splitlines()
    line_count = len(lines)
    if line_count == 0:
        return []
    caption_set = {cap.strip() for cap in captions if cap.strip()}
    if not caption_set:
        return []
    matchers: List[Tuple[str, str, Optional[str]]] = []
    for cap in caption_set:
        match = re.search(r"(table|figure)\s*(\d+)", cap, re.IGNORECASE)
        if match:
            matchers.append(("num", match.group(1).lower(), match.group(2)))
        else:
            matchers.append(("prefix", cap.lower()[:60], None))
    if not matchers:
        return []
    blocks: List[str] = []
    caption_re = re.compile(r"^\s*(table|figure)\s+\d+[:.]", re.IGNORECASE)
    section_heading_re = re.compile(r"^\s*\d+(?:\.\d+)*\.\s+\w")
    max_lines = 60
    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        line_lower = line_stripped.lower()
        matched = False
        for kind, token, num in matchers:
            if kind == "num":
                if re.search(rf"{re.escape(token)}\s*{re.escape(num)}\s*[:.]", line_lower):
                    matched = True
                    break
            else:
                if token and token in line_lower:
                    matched = True
                    break
        if not matched:
            continue
        start = idx
        collected: List[str] = []
        blank_run = 0
        row_count = 0
        for j in range(start, min(line_count, start + max_lines)):
            cur = lines[j].rstrip()
            if not cur.strip():
                blank_run += 1
            else:
                blank_run = 0
            if j > start and caption_re.match(cur):
                break
            if j > start and section_heading_re.match(cur):
                break
            if j > start and row_count >= 4:
                stripped = cur.strip()
                if (
                    len(stripped) > 80
                    and stripped[:1].isalpha()
                    and "." in stripped
                ):
                    break
                if (
                    stripped[:1].isalpha()
                    and not re.search(r"\d", stripped)
                    and len(stripped.split()) >= 6
                ):
                    break
            if blank_run >= 2:
                break
            if cur.strip() and cur.lstrip()[:1].isdigit():
                row_count += 1
            collected.append(cur)
        block = "\n".join(collected).strip()
        if block:
            blocks.append(block)
    return blocks


def extract_nearest_caption(lines: List[str], start_idx: int) -> Optional[str]:
    if start_idx <= 0:
        return None
    caption_re = re.compile(r"^\s*(table|figure|algorithm)\s+\d+[:.]", re.IGNORECASE)
    for i in range(start_idx - 1, max(-1, start_idx - 40), -1):
        line = lines[i].strip()
        if not line:
            continue
        if caption_re.match(line):
            return line
    return None


def has_blank_between(lines: List[str], start_idx: int, caption_idx: int) -> bool:
    if caption_idx >= start_idx:
        return False
    for i in range(caption_idx + 1, start_idx):
        if not lines[i].strip():
            return True
    return False


def extract_pdf_paragraphs(lines: List[str], start_idx: int, max_paragraphs: int = 2) -> Optional[str]:
    if start_idx <= 0:
        return None
    paragraphs: List[str] = []
    current: List[str] = []
    i = start_idx - 1
    while i >= 0 and len(paragraphs) < max_paragraphs:
        line = lines[i].strip()
        if not line:
            if current:
                paragraphs.append(" ".join(reversed(current)).strip())
                current = []
            i -= 1
            continue
        current.append(line)
        i -= 1
    if current and len(paragraphs) < max_paragraphs:
        paragraphs.append(" ".join(reversed(current)).strip())
    if not paragraphs:
        # Fallback: capture up to two non-empty lines above.
        fallback: List[str] = []
        i = start_idx - 1
        while i >= 0 and len(fallback) < 2:
            line = lines[i].strip()
            if line:
                fallback.append(line)
            i -= 1
        if not fallback:
            return None
        fallback.reverse()
        return "\n".join(fallback).strip()
    paragraphs.reverse()
    return "\n\n".join(paragraphs).strip()


def extract_nearest_cq_line(lines: List[str], start_idx: int) -> Optional[str]:
    cq_re = re.compile(r"\bCQ\d+\b", re.IGNORECASE)
    code_re = re.compile(r"\b(select|construct|ask|describe|where|prefix)\b", re.IGNORECASE)
    for i in range(start_idx - 1, max(-1, start_idx - 20), -1):
        line = lines[i].strip()
        if not line:
            continue
        if code_re.search(line):
            continue
        if cq_re.search(line) or line.endswith("?"):
            prev = None
            j = i - 1
            while j >= 0:
                prev_line = lines[j].strip()
                if not prev_line:
                    break
                if code_re.search(prev_line):
                    break
                prev = prev_line
                break
            if prev and (cq_re.search(prev) or line[:1].islower() or not cq_re.search(line)):
                return f"{prev} {line}".strip()
            return line
    return None


def extract_pdf_code_blocks(text: str) -> List[Dict[str, object]]:
    if not text.strip():
        return []
    lines = text.splitlines()
    blocks: List[Dict[str, object]] = []
    in_block = False
    current: List[str] = []
    start_idx = 0
    char_idx = 0
    start_char = 0
    for i, line in enumerate(lines):
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
                start_idx = i
                start_char = char_idx
                current = []
            current.append(line.rstrip())
            char_idx += len(line) + 1
            continue
        if in_block:
            if len(current) >= 3:
                blocks.append({"start_idx": start_idx, "start_char": start_char, "block": "\n".join(current).strip()})
            in_block = False
            current = []
        char_idx += len(line) + 1
    if in_block and len(current) >= 3:
        blocks.append({"start_idx": start_idx, "start_char": start_char, "block": "\n".join(current).strip()})
    return blocks


def extract_pdf_query_blocks(text: str) -> List[Dict[str, object]]:
    lines = text.splitlines()
    blocks: List[Dict[str, object]] = []
    current: List[str] = []
    in_block = False
    depth = 0
    seen_query = False
    start_idx = 0
    for idx, line in enumerate(lines):
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
                start_idx = idx
            current.append(line.rstrip())
            if re.search(r"\b(select|construct|ask|describe)\b", stripped, re.IGNORECASE):
                seen_query = True
            depth += line.count("{") - line.count("}")
            continue
        if in_block:
            depth += line.count("{") - line.count("}")
            current.append(line.rstrip())
            if seen_query and depth <= 0:
                if current:
                    blocks.append({"start_idx": start_idx, "lines": current[:]})
                current = []
                in_block = False
                depth = 0
                seen_query = False
                continue
    if in_block and current:
        blocks.append({"start_idx": start_idx, "lines": current[:]})

    # Normalize PREFIX lines / broken IRIs like extract_queries_from_pdf_text.
    normalized: List[Dict[str, object]] = []
    for block in blocks:
        merged: List[str] = []
        lines_block = block["lines"]
        i = 0
        while i < len(lines_block):
            line = lines_block[i].rstrip()
            if (
                line.strip().lower().startswith("prefix")
                and i + 1 < len(lines_block)
                and lines_block[i + 1].strip().startswith("<")
            ):
                line = f"{line} {lines_block[i + 1].strip()}"
                i += 2
                merged.append(line)
                continue
            if line.strip().endswith(":") and i + 1 < len(lines_block) and lines_block[i + 1].strip().startswith("<"):
                line = f"{line} {lines_block[i + 1].strip()}"
                i += 2
                merged.append(line)
                continue
            if "<" in line and ">" not in line and i + 1 < len(lines_block):
                next_line = lines_block[i + 1].strip()
                if next_line and not re.match(r"^\s*(prefix|select|construct|ask|describe)\b", next_line, re.IGNORECASE):
                    line = f"{line}{next_line}"
                    i += 2
                    merged.append(line)
                    continue
            merged.append(line)
            i += 1
        normalized.append({"start_idx": block["start_idx"], "block": "\n".join(merged).strip()})
    return normalized


def extract_pdf_cq_bullets(text: str) -> List[str]:
    lines = text.splitlines()
    bullets: List[str] = []
    header_re = re.compile(r"competency\\s+questions?", re.IGNORECASE)
    bullet_re = re.compile(r"^\\s*(?:[-*•]|\\d+\\.)\\s+")
    for i, line in enumerate(lines):
        if not header_re.search(line):
            continue
        # Look ahead for bullet list items.
        collected: List[str] = []
        for j in range(i + 1, min(i + 30, len(lines))):
            cur = lines[j].rstrip()
            if bullet_re.match(cur):
                collected.append(cur.strip())
                continue
            if collected and cur.strip() == "":
                break
            if collected and not bullet_re.match(cur):
                break
        if collected:
            bullets.append("\n".join(collected).strip())
    return bullets


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


def extract_cq_section(text: str) -> List[str]:
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
            if not snippet:
                return []
            return extract_cq_items_from_text(snippet)
    return []


def extract_bullet_items(lines: List[str]) -> List[str]:
    items: List[str] = []
    current: List[str] = []
    bullet_re = re.compile(r"^\s*[-*•]\s+(.*)")
    for line in lines:
        match = bullet_re.match(line)
        if match:
            if current:
                items.append(" ".join(current).strip())
                current = []
            current.append(match.group(1).strip())
            continue
        if current and line.strip():
            current.append(line.strip())
            continue
        if current:
            items.append(" ".join(current).strip())
            current = []
    if current:
        items.append(" ".join(current).strip())
    return [item for item in items if item]


def extract_cq_items_from_text(text: str) -> List[str]:
    items: List[str] = []
    label_items = extract_label_blocks(text)
    if label_items:
        items.extend(label_items)
        return items
    table_items = extract_table_blocks(text)
    if table_items:
        items.extend(table_items)
        return items
    bullet_items = extract_bullet_items(text.splitlines())
    if bullet_items:
        items.extend(bullet_items)
        return items
    for line in text.splitlines():
        if "question" in line.lower() or line.strip().lower().startswith("cq"):
            line = line.strip()
            items.extend(split_multi_cq_line(line))
    split_items: List[str] = []
    for item in items:
        split_items.extend(split_numbered_sequence(item))
    return [item for item in split_items if item]


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
                block.append(lines[i].rstrip())
                i += 1
            if block:
                results.extend(extract_bullet_items(block))
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
                for row in rows:
                    if row.strip():
                        blocks.append(row.strip())
                continue
        i += 1
    return blocks


def extract_cq_block(text: str) -> List[str]:
    pattern = re.compile(
        r"(#{1,6}\s+.*competency question.*?)(?=\n\s*#{1,6}\s+|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return []
    block = match.group(1)
    lines = block.splitlines()[1:]
    cleaned = "\n".join([ln.rstrip() for ln in lines if ln.strip()]).strip()
    if not cleaned:
        return []
    return extract_cq_items_from_text(cleaned)


def split_cq_block_items(block: str) -> List[str]:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return []
    if any("|" in ln for ln in lines):
        items: List[str] = []
        for ln in lines:
            if "|" not in ln:
                continue
            row = [c.strip() for c in ln.strip("|").split("|") if c.strip()]
            if row:
                items.append(" | ".join(row))
        split_items: List[str] = []
        for item in items:
            split_items.extend(split_numbered_sequence(item))
        return [item for item in split_items if item]
    label_re = re.compile(r"^(?:[A-Z]{2,3}\d+|CQ\d+|CT\d+|DR\d+|\d+)\b", re.IGNORECASE)
    items: List[str] = []
    current: List[str] = []
    for ln in lines:
        if label_re.match(ln):
            if current:
                items.append(" ".join(current).strip())
                current = []
            current.append(ln)
            continue
        if current:
            current.append(ln)
            continue
        items.append(ln)
    if current:
        items.append(" ".join(current).strip())
    split_items: List[str] = []
    for item in items:
        split_items.extend(split_numbered_sequence(item))
    return [item for item in split_items if item]


def split_numbered_sequence(text: str) -> List[str]:
    line = text.strip()
    if not line:
        return []
    matches = list(re.finditer(r"\b\d+\s+\w", line))
    if len(matches) < 2:
        return [line]
    parts: List[str] = []
    starts = [m.start() for m in matches]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(line)
        part = line[start:end].strip()
        if part:
            parts.append(part)
    return parts if parts else [line]


def split_multi_cq_line(line: str) -> List[str]:
    if not line:
        return []
    if len(re.findall(r"\bCQ\d+\b", line, flags=re.IGNORECASE)) < 2:
        return [line]
    parts = re.findall(r"(CQ\d+.*?)(?=\bCQ\d+\b|$)", line, flags=re.IGNORECASE)
    cleaned: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "?" in part:
            part = part[: part.rfind("?") + 1].strip()
        cleaned.append(part)
    return cleaned if cleaned else [line]


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


def query_has_pdf_evidence(rec: Dict[str, object], pdf_path: Path) -> bool:
    evidence = rec.get("evidence")
    if not isinstance(evidence, list):
        return False
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        if ev.get("source_path") != str(pdf_path):
            continue
        if ev.get("type") in {"doc_pre", "doc_fence", "doc_pdf"}:
            return True
    return False


def normalize_query_signature(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def query_has_doc_evidence(rec: Dict[str, object]) -> bool:
    for ev in rec.get("evidence", []) or []:
        if isinstance(ev, dict) and ev.get("type") in {"doc_pre", "doc_fence"}:
            return True
    return False


def rank_llm_context(evidence: List[Dict[str, object]]) -> List[Dict[str, object]]:
    priority_groups = [
        {"query_comment"},
        {"doc_query_desc", "web_query_desc", "readme_query_desc"},
        {"cq_item"},
        {"kg_summary", "doc_summary", "readme_summary", "web_summary", "repo_summary"},
    ]
    type_rank: Dict[str, int] = {}
    for idx, group in enumerate(priority_groups):
        for t in group:
            type_rank[t] = idx
    ranked: List[Dict[str, object]] = []
    for pos, ev in enumerate(evidence):
        ev_type = ev.get("type") if isinstance(ev, dict) else None
        rank = type_rank.get(ev_type, len(priority_groups))
        ranked.append((rank, pos, ev))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked]


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
        snippet = ev.get("snippet") or ""
        if not isinstance(snippet, str) or not snippet.strip():
            continue
        if ev.get("type") == "cq_item":
            cleaned = clean_desc(snippet)
            if not cleaned:
                continue
            if re.match(r"^\s*(table|figure|algorithm)\s+\d+[:.].*competency\s+questions", cleaned, re.IGNORECASE):
                continue
            ev = {**ev, "snippet": cleaned}
        key = (ev.get("type"), ev.get("source_path"), ev.get("snippet"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)
    return deduped


def expand_cq_items(evidence: List[Dict[str, object]]) -> List[Dict[str, object]]:
    expanded: List[Dict[str, object]] = []
    for ev in evidence:
        if not isinstance(ev, dict) or ev.get("type") != "cq_item":
            expanded.append(ev)
            continue
        snippet = ev.get("snippet") or ""
        if not isinstance(snippet, str) or not snippet.strip():
            continue
        parts = split_multi_cq_line(snippet)
        split_items: List[str] = []
        for part in parts:
            split_items.extend(split_numbered_sequence(part))
        if len(split_items) <= 1:
            expanded.append(ev)
            continue
        for part in split_items:
            if not part.strip():
                continue
            new_ev = dict(ev)
            new_ev["snippet"] = part.strip()
            expanded.append(new_ev)
    return expanded


def renumber_evidence(evidence: List[Dict[str, object]]) -> List[Dict[str, object]]:
    renumbered: List[Dict[str, object]] = []
    for idx, ev in enumerate(evidence, start=1):
        if not isinstance(ev, dict):
            continue
        ev = dict(ev)
        ev["evidence_id"] = f"e{idx}"
        renumbered.append(ev)
    return renumbered


def main() -> None:
    queries_path = Path("kg_queries.jsonl")
    repos_dir = Path("repos")
    sources_dir = Path("kg_sources")
    pdfs_dir = Path("pdfs")
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
            docs = kg.get("docs")
            if isinstance(kg_id, str) and isinstance(docs, list):
                for doc in docs:
                    if not isinstance(doc, str):
                        continue
                    doc_path = Path(doc)
                    if doc_path.exists():
                        kg_sources.setdefault(kg_id, [])
                        if str(doc_path) not in kg_sources[kg_id]:
                            kg_sources[kg_id].append(str(doc_path))

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
                    cq_items = []
                    heading_blocks = extract_heading_bullets(text)
                    if heading_blocks:
                        cq_items.extend(heading_blocks)
                    else:
                        cq_items.extend(extract_label_blocks(text))
                    table_blocks = extract_table_blocks(text)
                    if table_blocks:
                        cq_items.extend(table_blocks)
                    if not cq_items:
                        cq_items.extend(extract_cq_block(text))
                    for rec2 in records:
                        if rec2.get("kg_id") != kg_id:
                            continue
                        if query_has_doc_evidence(rec2):
                            continue
                        for item in cq_items:
                            add_evidence(
                                rec2,
                                "cq_item",
                                repo_url,
                                source_path,
                                str(repo_commit or ""),
                                clean_desc(item),
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
                cq_items = []
                heading_blocks = extract_heading_bullets(readme_text)
                if heading_blocks:
                    cq_items.extend(heading_blocks)
                else:
                    cq_items.extend(extract_label_blocks(readme_text))
                table_blocks = extract_table_blocks(readme_text)
                if table_blocks:
                    cq_items.extend(table_blocks)
                if not cq_items:
                    cq_items.extend(extract_cq_block(readme_text))
                for rec2 in records:
                    if rec2.get("kg_id") != kg_id:
                        continue
                    if query_has_doc_evidence(rec2):
                        continue
                    for item in cq_items:
                        add_evidence(
                            rec2,
                            "cq_item",
                            repo_url,
                            str(readme.relative_to(repo_dir)),
                            "",
                            clean_desc(item),
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
            cq_items = []
            heading_blocks = extract_heading_bullets(body)
            if heading_blocks:
                cq_items.extend(heading_blocks)
            else:
                cq_items.extend(extract_label_blocks(body))
            table_blocks = extract_table_blocks(body)
            if table_blocks:
                cq_items.extend(table_blocks)
            if not cq_items:
                cq_items.extend(extract_cq_block(body))
            for rec2 in records:
                if rec2.get("kg_id") != kg_id:
                    continue
                has_same_source = any(
                    isinstance(e, dict) and e.get("source_path") == str(src_path)
                    for e in rec2.get("evidence", []) or []
                )
                if not (has_same_source or query_has_repo_evidence(rec2)):
                    continue
                for item in cq_items:
                    add_evidence(
                        rec2,
                        "cq_item",
                        source_url or "",
                        str(src_path),
                        "",
                        clean_desc(item),
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

            cq_section_items = extract_cq_section(body)
            if cq_section_items and not doc_cq_seen and not table_blocks:
                for rec2 in records:
                    if rec2.get("kg_id") != kg_id:
                        continue
                    if not any(
                        isinstance(e, dict) and e.get("source_path") == str(src_path)
                        for e in rec2.get("evidence", []) or []
                    ):
                        continue
                    for item in cq_section_items:
                        add_evidence(
                            rec2,
                            "cq_item",
                            source_url or "",
                            str(src_path),
                            "",
                            clean_desc(item),
                            extracted_at,
                        )

        # Extract evidence from PDFs by filename match or explicit doc paths.
        pdf_paths: List[Path] = []
        if pdfs_dir.exists():
            pdf_paths.extend(pdfs_dir.glob("*.pdf"))
        for doc_path in kg_sources.get(kg_id, []):
            path_obj = Path(doc_path)
            if path_obj.suffix.lower() == ".pdf" and path_obj.exists():
                pdf_paths.append(path_obj)
        seen_pdfs: set[Path] = set()
        for pdf_path in pdf_paths:
            if pdf_path in seen_pdfs:
                continue
            seen_pdfs.add(pdf_path)
            if not isinstance(kg_id, str):
                continue
            if kg_id.lower() not in pdf_path.name.lower() and str(pdf_path) not in kg_sources.get(kg_id, []):
                continue
            # Remove previous PDF-derived CQ evidence so we can replace it cleanly.
            for rec2 in records:
                if rec2.get("kg_id") != kg_id:
                    continue
                ev = rec2.get("evidence")
                if not isinstance(ev, list):
                    continue
                rec2["evidence"] = [
                    e for e in ev
                    if not (
                        isinstance(e, dict)
                        and e.get("source_path") == str(pdf_path)
                        and e.get("type") in {"cq_item", "doc_query_desc"}
                    )
                ]
            pdf_text = extract_text_from_pdf(pdf_path)
            if not pdf_text.strip():
                continue
            captions = extract_pdf_captions(pdf_text) + extract_pdf_cq_captions(pdf_text)
            caption_hits = []
            for cap in captions:
                cap_lower = cap.lower()
                if "table" in cap_lower and re.search(r"competency\s+questions?", cap_lower):
                    caption_hits.append(cap)
            caption_hits = sorted(set(caption_hits))
            cq_table_blocks = extract_pdf_tables_for_captions(pdf_text, caption_hits) if caption_hits else []
            cq_bullets = extract_pdf_cq_bullets(pdf_text)
            code_blocks = extract_pdf_query_blocks(pdf_text)
            # Only fall back to generic CQ extraction when no explicit CQ signal was found.
            heading_blocks = [] if (caption_hits or cq_bullets) else extract_heading_bullets(pdf_text)
            label_blocks = [] if (caption_hits or cq_bullets) else extract_label_blocks(pdf_text)
            fallback_tables = [] if (caption_hits or cq_bullets) else extract_table_blocks(pdf_text)
            cq_section_items = [] if (caption_hits or cq_bullets) else extract_cq_section(pdf_text)
            pdf_lines = pdf_text.splitlines()
            for rec2 in records:
                if rec2.get("kg_id") != kg_id:
                    continue
                if not any(
                    isinstance(e, dict) and e.get("source_path") == str(pdf_path)
                    for e in rec2.get("evidence", []) or []
                ) and not query_has_repo_evidence(rec2):
                    continue
                for tbl in cq_table_blocks:
                    for item in split_cq_block_items(tbl):
                        add_evidence(
                            rec2,
                            "cq_item",
                            "",
                            str(pdf_path),
                            "",
                            clean_desc(item),
                            extracted_at,
                        )
                for bullet_block in cq_bullets:
                    for item in extract_bullet_items(bullet_block.splitlines()):
                        add_evidence(
                            rec2,
                            "cq_item",
                            "",
                            str(pdf_path),
                            "",
                            clean_desc(item),
                            extracted_at,
                        )
                for block in heading_blocks + label_blocks:
                    add_evidence(
                        rec2,
                        "cq_item",
                        "",
                        str(pdf_path),
                        "",
                        clean_desc(block),
                        extracted_at,
                    )
                for tbl in fallback_tables:
                    for item in split_cq_block_items(tbl):
                        add_evidence(
                            rec2,
                            "cq_item",
                            "",
                            str(pdf_path),
                            "",
                            clean_desc(item),
                            extracted_at,
                        )
                for item in cq_section_items:
                    add_evidence(
                        rec2,
                        "cq_item",
                        "",
                        str(pdf_path),
                        "",
                        clean_desc(item),
                        extracted_at,
                    )

            # Attach query descriptions to matching PDF-derived queries only.
            matched_targets: List[Dict[str, object]] = []
            pdf_targets: Dict[str, Dict[str, object]] = {}
            for rec2 in records:
                if rec2.get("kg_id") != kg_id:
                    continue
                if not query_has_pdf_evidence(rec2, pdf_path):
                    continue
                sparql_clean = rec2.get("sparql_clean")
                if isinstance(sparql_clean, str):
                    pdf_targets[normalize_query_signature(sparql_clean)] = rec2
            for block in code_blocks:
                for segment in split_queries_with_starts(block.get("block", "")):
                    raw_query = segment["query"]
                    normalized = normalize_query(raw_query)
                    if not normalized:
                        continue
                    target = pdf_targets.get(normalize_query_signature(normalized))
                    if target is None:
                        continue
                    start_char = int(block.get("start_char", 0))
                    start_idx = int(block.get("start_idx", 0))
                    caption = extract_nearest_caption(pdf_lines, start_idx)
                    caption_idx = None
                    if caption:
                        for i in range(start_idx - 1, max(-1, start_idx - 40), -1):
                            if pdf_lines[i].strip() == caption:
                                caption_idx = i
                                break
                    caption_is_near = caption_idx is not None and (start_idx - caption_idx) <= 3 and not has_blank_between(pdf_lines, start_idx, caption_idx)
                    if caption and caption_is_near:
                        caption_clean = clean_desc(caption)
                        if caption_clean:
                            add_evidence(
                                target,
                                "doc_query_desc",
                                "",
                                str(pdf_path),
                                "",
                                caption_clean,
                                extracted_at,
                            )
                    else:
                        cq_line = extract_nearest_cq_line(pdf_lines, start_idx)
                        if cq_line:
                            context_clean = clean_desc(cq_line)
                        else:
                            context = extract_pdf_paragraphs(pdf_lines, start_idx, max_paragraphs=2)
                            context_clean = clean_desc(context) if context else ""
                        if context_clean:
                            add_evidence(
                                target,
                                "doc_query_desc",
                                "",
                                str(pdf_path),
                                "",
                                context_clean,
                                extracted_at,
                            )
                    matched_targets.append(target)

    after_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    for rec in records:
        evidence = rec.get("evidence")
        if isinstance(evidence, list):
            rec["evidence"] = renumber_evidence(dedupe_evidence(expand_cq_items(evidence)))
            for ev in rec["evidence"]:
                if isinstance(ev, dict) and ev.get("type"):
                    type_counts[ev["type"]] = type_counts.get(ev["type"], 0) + 1
            kg_id = rec.get("kg_id")
            if isinstance(kg_id, str):
                after_counts[kg_id] = after_counts.get(kg_id, 0) + len(rec["evidence"])
            rec.pop("llm_context", None)
            rec.pop("llm_context_ranked", None)
            rec.pop("cq_items", None)
            rec.pop("justification", None)
            rec.pop("comments", None)
            if "confidence" not in rec:
                rec["confidence"] = None
            if "llm_output" not in rec:
                rec["llm_output"] = {
                    "ranked_evidence_phrases": [],
                    "nl_question": None,
                    "nl_question_origin": {
                        "mode": None,
                        "evidence_ids": [],
                        "primary_evidence_id": None,
                    },
                    "confidence": None,
                    "confidence_rationale": None,
                    "needs_review": None,
                }

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
