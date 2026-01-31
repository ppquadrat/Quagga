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


def extract_md_blocks_with_desc(text: str) -> List[Dict[str, str]]:
    pattern = re.compile(r"```sparql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    results: List[Dict[str, str]] = []
    matches = list(pattern.finditer(text))
    for match in matches:
        block = match.group(1)
        prefix = text[: match.start()]
        parts = re.split(r"\n\s*\n", prefix)
        desc = ""
        for part in reversed(parts):
            cleaned = clean_md_text(part)
            if cleaned:
                desc = cleaned
                break
        results.append({"query": block, "desc": desc})
    return results


def extract_pre_blocks_with_desc(text: str) -> List[Dict[str, str]]:
    pattern = re.compile(r"<pre>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
    results: List[Dict[str, str]] = []
    matches = list(pattern.finditer(text))
    for match in matches:
        block = html.unescape(match.group(1))
        prefix = text[: match.start()]
        parts = re.split(r"\n\s*\n", prefix)
        desc = ""
        for part in reversed(parts):
            cleaned = clean_md_text(part)
            if cleaned:
                desc = cleaned
                break
        results.append({"query": block, "desc": desc})
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


def parse_source_file(path: Path) -> Tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.startswith("SOURCE:"):
        parts = text.split("\n\n", 1)
        header = parts[0].strip()
        body = parts[1] if len(parts) > 1 else ""
        url = header.replace("SOURCE:", "").strip()
        return url, body
    return "", text


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


def extract_cq_list_from_markdown(text: str) -> Optional[str]:
    lines = text.splitlines()
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    start_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        lower = line.strip().lower()
        if "competency question" in lower or lower.startswith("cqs") or " cqs" in lower:
            start_idx = idx + 1
            break
    if start_idx is None:
        return None
    collected: List[str] = []
    for line in lines[start_idx:]:
        if heading_re.match(line):
            break
        if line.strip():
            collected.append(line.rstrip())
    snippet = "\n".join(collected).strip()
    return snippet if snippet else None


def extract_cq_block(text: str) -> Optional[str]:
    # Fallback: capture section from heading to next heading.
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


def select_llm_context(evidence: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def pick_by_types(types: List[str], limit: int) -> List[Dict[str, object]]:
        picked: List[Dict[str, object]] = []
        for ev in evidence:
            if ev.get("type") in types and ev.get("snippet"):
                picked.append(ev)
                if len(picked) >= limit:
                    break
        return picked

    context: List[Dict[str, object]] = []
    context.extend(pick_by_types(["query_comment"], 5))
    context.extend(pick_by_types(["doc_query_desc", "web_query_desc", "readme_query_desc"], 2))
    if not context:
        context.extend(pick_by_types(["paper_cq_section"], 1))
    return context[:6]


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
                        if desc:
                            add_evidence(
                                target,
                                "doc_query_desc",
                                repo_url,
                                source_path,
                                str(repo_commit or ""),
                                desc,
                                extracted_at,
                            )
                if file_path.name.lower().startswith("readme"):
                    cq_list = extract_cq_list_from_markdown(text) or extract_cq_block(text)
                    if cq_list:
                        for rec2 in records:
                            if rec2.get("kg_id") != kg_id:
                                continue
                            add_evidence(
                                rec2,
                                "cq_list",
                                repo_url,
                                source_path,
                                str(repo_commit or ""),
                                cq_list,
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
            cq_list = extract_cq_list_from_markdown(readme_text) or extract_cq_block(readme_text)
            if cq_list:
                for rec2 in records:
                    if rec2.get("kg_id") != kg_id:
                        continue
                    add_evidence(
                            rec2,
                            "cq_list",
                            repo_url,
                            str(readme.relative_to(repo_dir)),
                            "",
                            cq_list,
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
                        desc = block.get("desc", "")
                        if desc:
                            add_evidence(
                                target,
                                "readme_query_desc",
                                repo_url,
                                str(readme.relative_to(repo_dir)),
                                "",
                                desc,
                                extracted_at,
                            )

        # Enrich from kg_sources (web/papers) if available.
        source_files = kg_sources.get(kg_id, [])
        for src_file in source_files:
            src_path = Path(src_file)
            if not src_path.is_absolute():
                src_path = sources_dir / src_path
            if not src_path.exists():
                continue
            source_url, body = parse_source_file(src_path)
            if not body.strip():
                continue
            cq_list = extract_cq_list_from_markdown(body) or extract_cq_block(body)
            if cq_list:
                for rec2 in records:
                    if rec2.get("kg_id") != kg_id:
                        continue
                    add_evidence(
                        rec2,
                        "cq_list",
                        source_url or "",
                        str(src_path),
                        "",
                        cq_list,
                        extracted_at,
                    )
            # Try to match SPARQL blocks to queries.
            for block in extract_md_blocks_with_desc(body):
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
                    if desc:
                        add_evidence(
                            target,
                            "web_query_desc",
                            source_url or "",
                            str(src_path),
                            "",
                            desc,
                            extracted_at,
                        )

            cq_section = extract_cq_section(body)
            if cq_section:
                for rec2 in records:
                    if rec2.get("kg_id") != kg_id:
                        continue
                    add_evidence(
                        rec2,
                        "paper_cq_section",
                        source_url or "",
                        str(src_path),
                        "",
                        cq_section,
                        extracted_at,
                    )

    for rec in records:
        evidence = rec.get("evidence")
        if isinstance(evidence, list):
            rec["evidence"] = dedupe_evidence(evidence)
            rec["llm_context"] = select_llm_context(rec["evidence"])

    write_jsonl(queries_path, records)
    print(f"Wrote {len(records)} records to {queries_path.resolve()}")


if __name__ == "__main__":
    main()
