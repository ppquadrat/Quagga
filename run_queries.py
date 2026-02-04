#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yaml
from rdflib import Graph


@dataclass
class KGEndpoint:
    kg_id: str
    endpoint: str
    graph: Optional[str] = None


@dataclass
class KGDataset:
    kg_id: str
    dump_url: Optional[str]
    local_path: Optional[str]
    format: Optional[str]


def load_seeds(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing seeds file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("seeds.yaml must contain a top-level mapping (dictionary).")
    kgs = data.get("kgs")
    if not isinstance(kgs, list):
        raise ValueError("seeds.yaml must have a top-level key 'kgs' containing a list.")
    return kgs


def load_endpoints(path: Path) -> Dict[str, KGEndpoint]:
    endpoints: Dict[str, KGEndpoint] = {}
    for raw in load_seeds(path):
        kg_id = raw.get("kg_id")
        sparql = raw.get("sparql")
        if not isinstance(kg_id, str) or not kg_id.strip():
            continue
        if not isinstance(sparql, dict):
            continue
        endpoint = sparql.get("endpoint")
        graph = sparql.get("graph")
        if isinstance(endpoint, str) and endpoint.strip():
            graph_val = graph.strip() if isinstance(graph, str) and graph.strip() else None
            endpoints[kg_id.strip()] = KGEndpoint(
                kg_id=kg_id.strip(),
                endpoint=endpoint.strip(),
                graph=graph_val,
            )
    return endpoints


def load_datasets(path: Path) -> Dict[str, KGDataset]:
    datasets: Dict[str, KGDataset] = {}
    for raw in load_seeds(path):
        kg_id = raw.get("kg_id")
        dataset = raw.get("dataset")
        if not isinstance(kg_id, str) or not kg_id.strip():
            continue
        if not isinstance(dataset, dict):
            continue
        dump_url = dataset.get("dump_url")
        local_path = dataset.get("local_path")
        fmt = dataset.get("format")
        if dump_url is not None and not isinstance(dump_url, str):
            continue
        if local_path is not None and not isinstance(local_path, str):
            continue
        if fmt is not None and not isinstance(fmt, str):
            continue
        datasets[kg_id.strip()] = KGDataset(
            kg_id=kg_id.strip(),
            dump_url=dump_url.strip() if isinstance(dump_url, str) else None,
            local_path=local_path.strip() if isinstance(local_path, str) else None,
            format=fmt.strip() if isinstance(fmt, str) else None,
        )
    return datasets


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


def parse_sparql_xml(xml_text: str) -> Optional[Dict[str, object]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    ns = {"sr": "http://www.w3.org/2005/sparql-results#"}
    results = root.find("sr:results", ns)
    if results is None:
        return None

    bindings = []
    for result in results.findall("sr:result", ns):
        row: Dict[str, str] = {}
        for binding in result.findall("sr:binding", ns):
            name = binding.get("name")
            if not name:
                continue
            value_el = next(iter(binding), None)
            if value_el is None:
                continue
            row[name] = value_el.text or ""
        bindings.append(row)

    count = len(bindings)
    sample = bindings[0] if bindings else None
    return {"status": "ok" if count > 0 else "empty", "result_count": count, "sample_row": sample}


def extract_html_error(html_text: str) -> str:
    match = re.search(r"<h1>\\s*([^<]+)\\s*</h1>", html_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"<p>\\s*([^<]+)\\s*</p>", html_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()[:200]


def extract_html_error_line(html_text: str) -> str:
    match = re.search(r"<h1>\\s*([^<]+)\\s*</h1>", html_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"<p>\\s*([^<]+)\\s*</p>", html_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()[:120]


def run_select_query(endpoint: str, query: str, timeout_s: int = 30) -> Dict[str, object]:
    query = query.lstrip("\ufeff").strip()
    headers = {
        "Accept": "application/sparql-results+json, application/sparql-results+xml;q=0.9",
        "User-Agent": "kg-pipeline/0.1",
    }
    data = {"query": query, "format": "application/sparql-results+json"}
    raw_headers = headers | {"Content-Type": "application/sparql-query"}
    try:
        resp = requests.post(endpoint, data=data, headers=headers, timeout=timeout_s)
    except requests.RequestException as e:
        return {"status": "request_error", "error": f"{e.__class__.__name__}"}

    if resp.status_code != 200:
        # Some endpoints expect application/sparql-query POST bodies.
        try:
            resp = requests.post(endpoint, data=query.encode("utf-8"), headers=raw_headers, timeout=timeout_s)
        except requests.RequestException as e:
            return {"status": "request_error", "error": f"{e.__class__.__name__}"}
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type.lower() or "application/sparql-results+json" in content_type.lower():
                try:
                    payload = resp.json()
                except ValueError:
                    payload = None
                if payload is not None:
                    results = payload.get("results", {})
                    bindings = results.get("bindings", [])
                    if isinstance(bindings, list):
                        count = len(bindings)
                        sample = bindings[0] if bindings else None
                        return {"status": "ok" if count > 0 else "empty", "result_count": count, "sample_row": sample}
            xml_result = parse_sparql_xml(resp.text)
            if xml_result is not None:
                return xml_result
        # Some endpoints expect GET instead of POST, but avoid GET for long queries.
        if len(query) < 1500:
            try:
                resp = requests.get(endpoint, params=data, headers=headers, timeout=timeout_s)
            except requests.RequestException as e:
                return {"status": "request_error", "error": f"{e.__class__.__name__}"}
        if resp.status_code != 200:
            content_type = resp.headers.get("Content-Type", "")
            snippet = resp.text[:500] if resp.text else ""
            if "text/html" in content_type.lower() and resp.text:
                error_msg = extract_html_error(resp.text)
                error_line = extract_html_error_line(resp.text)
                status = "parse_error" if "parseexception" in error_msg.lower() or "parseexception" in error_line.lower() else "query_error"
                return {
                    "status": status,
                    "http_status": resp.status_code,
                    "content_type": content_type,
                    "error": error_msg,
                    "error_line": error_line,
                    "body_snippet": snippet,
                }
            return {
                "status": "http_error",
                "http_status": resp.status_code,
                "content_type": content_type,
                "body_snippet": snippet,
            }

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type.lower() and resp.text:
        # Try application/sparql-query POST as a fallback before classifying as query_error.
        try:
            alt = requests.post(endpoint, data=query.encode("utf-8"), headers=raw_headers, timeout=timeout_s)
        except requests.RequestException:
            alt = None
        if alt is not None and alt.status_code == 200:
            alt_type = alt.headers.get("Content-Type", "")
            if "application/json" in alt_type.lower() or "application/sparql-results+json" in alt_type.lower():
                try:
                    payload = alt.json()
                except ValueError:
                    payload = None
                if payload is not None:
                    results = payload.get("results", {})
                    bindings = results.get("bindings", [])
                    if isinstance(bindings, list):
                        count = len(bindings)
                        sample = bindings[0] if bindings else None
                        return {"status": "ok" if count > 0 else "empty", "result_count": count, "sample_row": sample}
            xml_result = parse_sparql_xml(alt.text)
            if xml_result is not None:
                return xml_result
    try:
        payload = resp.json()
    except ValueError:
        xml_result = parse_sparql_xml(resp.text)
        if xml_result is not None:
            return xml_result
        snippet = resp.text[:500] if resp.text else ""
        if "text/html" in content_type.lower() and resp.text:
            error_msg = extract_html_error(resp.text)
            error_line = extract_html_error_line(resp.text)
            status = "parse_error" if "parseexception" in error_msg.lower() or "parseexception" in error_line.lower() else "query_error"
            return {
                "status": status,
                "http_status": resp.status_code,
                "content_type": content_type,
                "error": error_msg,
                "error_line": error_line,
                "body_snippet": snippet,
            }
        return {"status": "bad_json", "content_type": content_type, "body_snippet": snippet}

    results = payload.get("results", {})
    bindings = results.get("bindings", [])
    if isinstance(bindings, list):
        count = len(bindings)
        sample = bindings[0] if bindings else None
        return {"status": "ok" if count > 0 else "empty", "result_count": count, "sample_row": sample}
    return {"status": "bad_results"}


def guess_rdf_format(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    return {
        ".ttl": "turtle",
        ".nt": "nt",
        ".nq": "nquads",
        ".rdf": "xml",
        ".xml": "xml",
        ".jsonld": "json-ld",
    }.get(ext)


def ensure_dump_available(dataset: KGDataset, limit_mb: int = 550) -> Path:
    if dataset.local_path:
        local_path = Path(dataset.local_path)
    else:
        local_path = Path("dumps") / f"{dataset.kg_id}.ttl"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        head = local_path.read_bytes()[:1024].lstrip()
        if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
            local_path.unlink(missing_ok=True)
        else:
            return local_path
    if not dataset.dump_url:
        raise FileNotFoundError(f"No dump_url for {dataset.kg_id} and local_path not found.")

    try:
        head = requests.head(dataset.dump_url, allow_redirects=True, timeout=20)
        size = head.headers.get("Content-Length")
        if size and int(size) > limit_mb * 1024 * 1024:
            raise ValueError(f"Dump too large ({size} bytes) for {dataset.kg_id}.")
    except requests.RequestException:
        pass

    max_bytes = limit_mb * 1024 * 1024
    downloaded = 0
    with requests.get(dataset.dump_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError(f"Dump exceeds limit ({limit_mb} MB) for {dataset.kg_id}.")
                f.write(chunk)
    head = local_path.read_bytes()[:1024].lstrip()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        local_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded HTML instead of RDF for {dataset.kg_id}. Check dump_url.")
    return local_path


def run_local_select_query(graph: Graph, query: str) -> Dict[str, object]:
    try:
        result = graph.query(query)
    except Exception as e:
        return {"status": "query_error", "error": f"{e.__class__.__name__}: {e}"}
    rows = list(result)
    count = len(rows)
    sample = None
    if rows:
        try:
            sample = {k: str(v) for k, v in rows[0].asdict().items()}
        except Exception:
            sample = None
    return {"status": "ok" if count > 0 else "empty", "result_count": count, "sample_row": sample}


def clean_query(query: str) -> str:
    # Drop leading/trailing comment-only lines to avoid endpoint parser quirks.
    lines = query.splitlines()
    start = 0
    while start < len(lines) and (not lines[start].strip() or lines[start].lstrip().startswith("#")):
        start += 1
    end = len(lines)
    while end > start and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
        end -= 1
    return "\n".join(lines[start:end]).strip()


def ensure_prefixes(query: str) -> str:
    prefix_map = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "dc": "http://purl.org/dc/elements/1.1/",
        "dtl": "http://www.DTL.org/schema/properties/",
        "event": "http://purl.org/NET/c4dm/event.owl#",
        "mo": "http://purl.org/ontology/mo/",
        "tl": "http://purl.org/NET/c4dm/timeline.owl#",
        "foaf": "http://xmlns.com/foaf/0.1/",
    }
    normalized_query = query
    for prefix in prefix_map:
        normalized_query = re.sub(rf"\b{prefix.upper()}:", f"{prefix}:", normalized_query)
    existing = {m.group(1).lower() for m in re.finditer(r"(?im)^\\s*prefix\\s+(\\w+):", normalized_query)}
    needed: List[str] = []
    for prefix, iri in prefix_map.items():
        if prefix in existing:
            continue
        if re.search(rf"\b{re.escape(prefix)}:", normalized_query):
            needed.append(f"PREFIX {prefix}: <{iri}>")
    if needed:
        return "\n".join(needed) + "\n" + normalized_query
    return normalized_query


def apply_graph(query: str, graph: Optional[str]) -> str:
    if not graph:
        return query
    # Avoid rewriting if the query already declares a dataset.
    if re.search(r"(?im)^\\s*from\\b", query):
        return query
    lines = query.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if re.match(r"(?im)^\\s*(prefix|base)\\b", line):
            insert_at = i + 1
        elif line.strip() and not line.lstrip().startswith("#"):
            break
    lines.insert(insert_at, f"FROM <{graph}>")
    return "\n".join(lines)


def is_remote_executable(query: str) -> bool:
    lowered = query.lower()
    if "x-sparql-anything" in lowered:
        return False
    if "fx:" in lowered and "sparql.xyz/facade-x" in lowered:
        return False
    if "file://" in lowered:
        return False
    return True


def preflight_endpoint(endpoint: KGEndpoint) -> None:
    probe_default = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
    probe_named = "SELECT ?g WHERE { GRAPH ?g { ?s ?p ?o } } LIMIT 1"
    default_res = run_select_query(endpoint.endpoint, probe_default, timeout_s=20)
    named_res = run_select_query(endpoint.endpoint, probe_named, timeout_s=20)
    sample = named_res.get("sample_row") or {}
    graph_uri = None
    if isinstance(sample, dict):
        graph_val = sample.get("g")
        if isinstance(graph_val, dict):
            graph_uri = graph_val.get("value")
        elif isinstance(graph_val, str):
            graph_uri = graph_val

    if (
        named_res.get("status") == "ok"
        and isinstance(graph_uri, str)
        and endpoint.kg_id.lower() not in graph_uri.lower()
    ):
        print(
            f"warning: {endpoint.kg_id} graph name mismatch. "
            f"sample graph {graph_uri} does not include kg_id."
        )

    if default_res.get("status") == "empty" and named_res.get("status") == "ok":
        if not endpoint.graph:
            msg = f"warning: {endpoint.kg_id} default graph empty; named graph data found"
            if graph_uri:
                msg += f" (e.g. {graph_uri})"
            msg += ". Consider setting sparql.graph in seeds.yaml."
            print(msg)
        elif graph_uri and endpoint.graph not in graph_uri and endpoint.graph != graph_uri:
            print(
                f"warning: {endpoint.kg_id} graph mismatch. "
                f"configured={endpoint.graph}, sample={graph_uri}"
            )


def is_endpoint_healthy(endpoint: KGEndpoint) -> bool:
    probe = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
    res = run_select_query(endpoint.endpoint, probe, timeout_s=20)
    if res.get("status") in {"ok", "empty"}:
        return True
    if res.get("status") in {"bad_json", "http_error", "request_error"}:
        return False
    return False


def main() -> None:
    seeds_path = Path("seeds.yaml")
    queries_path = Path("kg_queries.jsonl")
    out_path = queries_path
    fail_path = Path("runnable_queries.failures.jsonl")

    endpoints = load_endpoints(seeds_path)
    datasets = load_datasets(seeds_path)
    raw_queries = load_query_records(queries_path)

    unhealthy_endpoints = set()
    for endpoint in endpoints.values():
        preflight_endpoint(endpoint)
        if not is_endpoint_healthy(endpoint):
            unhealthy_endpoints.add(endpoint.kg_id)
            print(f"warning: {endpoint.kg_id} endpoint appears unavailable; skipping queries.")

    records: List[Dict[str, object]] = raw_queries
    skipped_no_endpoint = 0
    kept = 0
    failures: List[Dict[str, object]] = []
    endpoint_success: Dict[str, int] = {}
    stats: Dict[str, Dict[str, int]] = {}
    graphs: Dict[str, Graph] = {}
    max_dump_mb = int(os.environ.get("KG_DUMP_MAX_MB", "550"))
    totals: Dict[str, int] = {}
    for rec in raw_queries:
        kg_id = rec.get("kg_id")
        if isinstance(kg_id, str):
            totals[kg_id] = totals.get(kg_id, 0) + 1
    processed: Dict[str, int] = {}
    current_kg = None
    for rec in raw_queries:
        kg_id = rec.get("kg_id")
        query = rec.get("sparql_clean")
        if not isinstance(kg_id, str) or not isinstance(query, str):
            continue
        if current_kg != kg_id:
            current_kg = kg_id
            print(f"\nRunning queries for {kg_id} ({totals.get(kg_id, 0)} total)")
        processed[kg_id] = processed.get(kg_id, 0) + 1
        label = rec.get("query_label") or rec.get("query_id")
        print(f"[{kg_id}] {processed[kg_id]}/{totals.get(kg_id, 0)} {label}")
        stat = stats.setdefault(
            kg_id,
            {
                "attempted": 0,
                "ran": 0,
                "ok": 0,
                "empty": 0,
                "failed": 0,
                "skipped_no_endpoint": 0,
                "skipped_local": 0,
            },
        )
        stat["attempted"] += 1
        endpoint = endpoints.get(kg_id)
        dataset = datasets.get(kg_id)
        if endpoint is not None and kg_id in unhealthy_endpoints:
            failures.append(
                {
                    "kg_id": kg_id,
                    "endpoint": endpoint.endpoint,
                    "status": "skipped_endpoint_unavailable",
                    "query_id": rec.get("query_id"),
                    "query_label": rec.get("query_label"),
                    "sparql_hash": rec.get("sparql_hash"),
                }
            )
            stat["skipped_local"] += 1
            rec["latest_run"] = {
                "ran_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "status": "skipped_endpoint_unavailable",
                "endpoint": endpoint.endpoint,
                "result_count": None,
                "sample_row": None,
                "duration_ms": 0,
                "error": None,
            }
            continue
        if endpoint is None and dataset is None:
            skipped_no_endpoint += 1
            stat["skipped_no_endpoint"] += 1
            continue

        query_to_run = clean_query(query)
        query_to_run = ensure_prefixes(query_to_run)
        if endpoint is not None:
            query_to_run = apply_graph(query_to_run, endpoint.graph)
            if not is_remote_executable(query_to_run):
                failures.append(
                    {
                        "kg_id": kg_id,
                        "endpoint": endpoint.endpoint,
                        "status": "skipped_local_query",
                        "query_id": rec.get("query_id"),
                        "query_label": rec.get("query_label"),
                        "sparql_hash": rec.get("sparql_hash"),
                    }
                )
                stat["skipped_local"] += 1
                rec["latest_run"] = {
                    "ran_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "status": "skipped_local_query",
                    "endpoint": endpoint.endpoint,
                    "result_count": None,
                    "sample_row": None,
                    "duration_ms": 0,
                    "error": None,
                }
                continue
        stat["ran"] += 1
        start = time.time()
        if endpoint is not None:
            result = run_select_query(endpoint.endpoint, query_to_run)
        else:
            dump_path = ensure_dump_available(dataset, limit_mb=max_dump_mb)
            if kg_id not in graphs:
                fmt = dataset.format or guess_rdf_format(dump_path)
                graph = Graph()
                graph.parse(dump_path, format=fmt)
                graphs[kg_id] = graph
            result = run_local_select_query(graphs[kg_id], query_to_run)
        duration_ms = int((time.time() - start) * 1000)
        if result.get("status") in {"ok", "empty"}:
            endpoint_success[kg_id] = endpoint_success.get(kg_id, 0) + 1
        elif endpoint is not None and result.get("status") == "http_error" and result.get("http_status") == 500:
            if endpoint_success.get(kg_id, 0) > 0:
                # Retry once or twice if the endpoint works for other queries.
                for delay_s in (1.0, 2.0):
                    time.sleep(delay_s)
                    retry_start = time.time()
                    retry = run_select_query(endpoint.endpoint, query_to_run)
                    duration_ms = int((time.time() - retry_start) * 1000)
                    if retry.get("status") in {"ok", "empty"}:
                        result = retry
                        endpoint_success[kg_id] = endpoint_success.get(kg_id, 0) + 1
                        break
                    result = retry
        status = result.get("status")
        latest_run = {
            "ran_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": status,
            "endpoint": endpoint.endpoint if endpoint is not None else None,
            "result_count": result.get("result_count"),
            "sample_row": result.get("sample_row"),
            "duration_ms": duration_ms,
            "error": result.get("error"),
        }
        if result.get("error_line") is not None:
            latest_run["error_line"] = result.get("error_line")
        if result.get("http_status") is not None:
            latest_run["http_status"] = result.get("http_status")
        if result.get("content_type") is not None:
            latest_run["content_type"] = result.get("content_type")
        rec["latest_run"] = latest_run
        run_history = rec.get("run_history")
        if not isinstance(run_history, list):
            run_history = []
        run_history.append(latest_run)
        rec["run_history"] = run_history
        if status not in {"ok", "empty"}:
            failures.append(
                {
                    "kg_id": kg_id,
                    "endpoint": endpoint.endpoint if endpoint is not None else None,
                    "status": status,
                    "http_status": result.get("http_status"),
                    "content_type": result.get("content_type"),
                    "error": result.get("error"),
                    "error_line": result.get("error_line"),
                    "query_id": rec.get("query_id"),
                    "query_label": rec.get("query_label"),
                    "sparql_hash": rec.get("sparql_hash"),
                    "body_snippet": result.get("body_snippet"),
                }
            )
            stat["failed"] += 1
            continue
        if status == "ok":
            stat["ok"] += 1
        else:
            stat["empty"] += 1
        rec["latest_successful_run"] = latest_run
        kept += 1

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with fail_path.open("w", encoding="utf-8") as f:
        for rec in failures:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if stats:
        print("\nPer-KG run stats:")
        for kg_id in sorted(stats):
            stat = stats[kg_id]
            attempted = stat["attempted"]
            ran = stat["ran"]
            runnable = stat["ok"] + stat["empty"]
            print(
                f"- {kg_id}: runnable {runnable}/{attempted} (ran={ran}, "
                f"ok={stat['ok']}, empty={stat['empty']}, failed={stat['failed']}, "
                f"skipped_no_endpoint={stat['skipped_no_endpoint']}, "
                f"skipped_local={stat['skipped_local']})"
            )
        if failures:
            error_lines: Dict[str, int] = {}
            for failure in failures:
                line = failure.get("error_line")
                if not isinstance(line, str) or not line.strip():
                    continue
                error_lines[line.strip()] = error_lines.get(line.strip(), 0) + 1
            if error_lines:
                print("\nTop error lines:")
                for line, count in sorted(error_lines.items(), key=lambda item: item[1], reverse=True)[:10]:
                    print(f"- {count}x {line}")
    print(
        f"Wrote {len(records)} records to {out_path.resolve()} "
        f"(skipped_no_endpoint={skipped_no_endpoint}, kept={kept}, failed={len(failures)})"
    )


if __name__ == "__main__":
    main()
