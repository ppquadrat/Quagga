#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yaml


@dataclass
class KGEndpoint:
    kg_id: str
    endpoint: str
    graph: Optional[str] = None


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


def run_select_query(endpoint: str, query: str, timeout_s: int = 30) -> Dict[str, object]:
    query = query.lstrip("\ufeff").strip()
    headers = {
        "Accept": "application/sparql-results+json, application/sparql-results+xml;q=0.9",
        "User-Agent": "kg-pipeline/0.1",
    }
    data = {"query": query, "format": "application/sparql-results+json"}
    try:
        resp = requests.post(endpoint, data=data, headers=headers, timeout=timeout_s)
    except requests.RequestException as e:
        return {"status": "request_error", "error": f"{e.__class__.__name__}"}

    if resp.status_code != 200:
        # Some endpoints expect GET instead of POST, but avoid GET for long queries.
        if len(query) < 1500:
            try:
                resp = requests.get(endpoint, params=data, headers=headers, timeout=timeout_s)
            except requests.RequestException as e:
                return {"status": "request_error", "error": f"{e.__class__.__name__}"}
        if resp.status_code != 200:
            snippet = resp.text[:500] if resp.text else ""
            return {
                "status": "http_error",
                "http_status": resp.status_code,
                "content_type": resp.headers.get("Content-Type", ""),
                "body_snippet": snippet,
            }

    content_type = resp.headers.get("Content-Type", "")
    try:
        payload = resp.json()
    except ValueError:
        xml_result = parse_sparql_xml(resp.text)
        if xml_result is not None:
            return xml_result
        snippet = resp.text[:500] if resp.text else ""
        return {"status": "bad_json", "content_type": content_type, "body_snippet": snippet}

    results = payload.get("results", {})
    bindings = results.get("bindings", [])
    if isinstance(bindings, list):
        count = len(bindings)
        sample = bindings[0] if bindings else None
        return {"status": "ok" if count > 0 else "empty", "result_count": count, "sample_row": sample}
    return {"status": "bad_results"}


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
    if "rdf:" in query and "prefix rdf:" not in query.lower():
        prefix = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"
        return f"{prefix}\n{query}"
    return query


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


def main() -> None:
    seeds_path = Path("seeds.yaml")
    queries_path = Path("kg_queries.jsonl")
    out_path = queries_path
    fail_path = Path("runnable_queries.failures.jsonl")

    endpoints = load_endpoints(seeds_path)
    raw_queries = load_query_records(queries_path)

    for endpoint in endpoints.values():
        preflight_endpoint(endpoint)

    records: List[Dict[str, object]] = raw_queries
    skipped_no_endpoint = 0
    kept = 0
    failures: List[Dict[str, object]] = []
    endpoint_success: Dict[str, int] = {}
    stats: Dict[str, Dict[str, int]] = {}
    for rec in raw_queries:
        kg_id = rec.get("kg_id")
        query = rec.get("sparql_clean")
        if not isinstance(kg_id, str) or not isinstance(query, str):
            continue
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
        if endpoint is None:
            skipped_no_endpoint += 1
            stat["skipped_no_endpoint"] += 1
            continue

        query_to_run = clean_query(query)
        query_to_run = ensure_prefixes(query_to_run)
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
        result = run_select_query(endpoint.endpoint, query_to_run)
        duration_ms = int((time.time() - start) * 1000)
        if result.get("status") in {"ok", "empty"}:
            endpoint_success[kg_id] = endpoint_success.get(kg_id, 0) + 1
        elif result.get("status") == "http_error" and result.get("http_status") == 500:
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
            "endpoint": endpoint.endpoint,
            "result_count": result.get("result_count"),
            "sample_row": result.get("sample_row"),
            "duration_ms": duration_ms,
            "error": result.get("error"),
        }
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
                    "endpoint": endpoint.endpoint,
                    "status": status,
                    "http_status": result.get("http_status"),
                    "content_type": result.get("content_type"),
                    "error": result.get("error"),
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
    print(
        f"Wrote {len(records)} records to {out_path.resolve()} "
        f"(skipped_no_endpoint={skipped_no_endpoint}, kept={kept}, failed={len(failures)})"
    )


if __name__ == "__main__":
    main()
