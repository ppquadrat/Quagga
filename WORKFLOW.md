# Workflow: Collecting NL–SPARQL Pairs for Musicological Knowledge Graphs

This repository implements a **reproducible pipeline for collecting, testing, and curating natural-language question–SPARQL query pairs** for musicological Knowledge Graphs (KGs). The resulting dataset is designed for evaluation, benchmarking, and ingestion into systems such as **Quagga**.

The workflow deliberately separates **control**, **deterministic processing**, and **LLM-assisted interpretation** to ensure auditability and long-term maintainability.

---

## 0. Design principles

- **YAML = control plane**  
  Defines what to process and where to find it. No results, no prose.

- **JSONL = curated outputs**  
  One record per line for KGs and NL–SPARQL pairs. Easy to diff, stream, and transform.

- **Python = truth layer**  
  Repo cloning, SPARQL execution, timeouts, provenance, filtering.

- **LLMs = language and interpretation layer**  
  KG descriptions, natural-language questions, confidence estimates.

This separation avoids hidden state, supports regeneration, and keeps the dataset defensible.

---

## 1. Seed definition (`seeds.yaml`)

**Purpose:** define which KGs to process and where their technical resources live.

Each KG entry typically includes:

- `kg_id` (stable identifier)
- human-readable name
- short `description_hint` (prompt hint, not authoritative)
- SPARQL endpoint (if available)
- repository URLs (one or more)
- optional documentation links
- priority and notes

Example:

    kgs:
      - kg_id: meetups
        name: Polifonia MEETUPS Knowledge Graph
        project: Polifonia
        description_hint: >
          Musical encounters and collaborations extracted from
          musician biographies (c. 1800–1945).
        sparql:
          endpoint: https://polifonia.kmi.open.ac.uk/meetups/sparql
          auth: none
        repos:
          - https://github.com/polifonia-project/meetups-kg

`seeds.yaml` is version-controlled and changes infrequently.

---

## 1.5. Data model (schemas)

To keep provenance and QA explicit, we use **two JSONL files**:

- `kgs.jsonl`: one record per KG (metadata, endpoints, datasets)
- `kg_queries.jsonl`: one record per query (SPARQL, evidence, NL artifacts)
- `run_queries.jsonl`: same records enriched with run metadata

### `kgs.jsonl` (KG metadata)

Each line is a JSON object. Example:

    {
      "kg_id": "meetups",
      "name": "Polifonia MEETUPS Knowledge Graph",
      "project": "Polifonia",
      "description": "...authoritative KG summary...",
      "sparql": {
        "endpoint": "https://polifonia.kmi.open.ac.uk/meetups/sparql",
        "auth": "none",
        "graph": null
      },
      "dataset": {
        "dump_url": null,
        "local_path": null,
        "format": null
      },
      "repos": ["https://github.com/polifonia-project/meetups-knowledge-graph"],
      "docs": ["https://polifonia.kmi.open.ac.uk/meetups/queries.php"],
      "notes": "...",
      "created_at": "2026-01-30",
      "updated_at": "2026-01-30"
    }

### `kg_queries.jsonl` (query-centric record)

One record per query, with provenance and run history:

    {
      "query_id": "musow__sha256:abc123...",
      "kg_id": "musow",
      "query_type": "select",
      "sparql_raw": "...as extracted...",
      "sparql_clean": "...normalized...",
      "sparql_hash": "sha256:...clean...",
      "raw_hash": "sha256:...raw...",
      "evidence": [
        {
          "evidence_id": "e1",
          "type": "repo_file",
          "source_url": "https://github.com/.../README.md",
          "source_path": "docs/queries.md",
          "repo_commit": "abc123",
          "snippet": "SELECT ...",
          "extracted_at": "2026-01-30",
          "extractor_version": "extract_queries.py@v1"
        }
      ],
      "confidence": null,
      "llm_output": {
        "ranked_evidence_phrases": [],
        "nl_question": null,
        "nl_question_origin": {
          "mode": null,
          "evidence_ids": [],
          "primary_evidence_id": null
        },
        "confidence": null,
        "confidence_rationale": null,
        "needs_review": null
      },
      "nl_question": {
        "text": null,
        "source": null,
        "generated_at": null,
        "generator": null
      },
      "verification": {
        "status": "unverified",
        "notes": null
      },
      "latest_run": {
        "ran_at": "2026-01-30T12:10:00Z",
        "status": "http_error",
        "endpoint": "https://polifonia.disi.unibo.it/meetups/sparql",
        "result_count": null,
        "sample_row": null,
        "duration_ms": 1820,
        "error": "http_500"
      },
      "latest_successful_run": {
        "ran_at": "2026-01-29T10:40:00Z",
        "status": "ok",
        "endpoint": "https://polifonia.disi.unibo.it/meetups/sparql",
        "result_count": 14,
        "sample_row": {"s": "..."},
        "duration_ms": 1200
      },
      "run_history": [
        {
          "ran_at": "2026-01-29T10:40:00Z",
          "status": "ok",
          "endpoint": "https://...",
          "duration_ms": 1200
        }
      ]
    }

Notes:

- `evidence` is the single place for raw extractions (repos/docs/papers/etc).
- CQ items are stored as `evidence` entries with `type: cq_item`.
- `confidence` is a combined score (LLM confidence + runnability + heuristics).
- `llm_output` stores the generated NL question, provenance, and LLM confidence.
- Machine-checkable schema for `llm_output`: `schemas/llm_output.schema.json`.
- `latest_run` and `latest_successful_run` are convenience fields; `run_history` is optional.
- These run-related fields are populated when producing `run_queries.jsonl`.
- `dataset` supports future KGs without endpoints (local dumps).

---

## 2. KG description generation (`kgs.jsonl`)

**Goal:** produce Quagga-ready KG records with rich, citable descriptions.

### Inputs

- `seeds.yaml`
- KG README files
- project websites
- related academic papers (abstracts or introductions)

### Process

For each KG:

1. Collect textual sources.
2. Use an LLM to generate a **120–180 word descriptive paragraph** suitable for a KG catalogue.
3. Record provenance (URLs and papers used).

### Output

`kgs.jsonl`, one KG per line, for example:

    {
      "kg_id": "meetups",
      "name": "Polifonia MEETUPS Knowledge Graph",
      "description": "...",
      "sparql": {
        "endpoint": "https://...",
        "auth": "none",
        "graph": null
      },
      "repos": ["https://github.com/..."],
      "description_sources": [
        "https://github.com/...",
        "Paper DOI ..."
      ]
    }

This file contains the **authoritative KG descriptions**.

---

## 3. SPARQL query extraction (`kg_queries.jsonl`)

**Goal:** collect all candidate SPARQL queries with full provenance, without interpretation.

### Inputs

- repositories listed in `seeds.yaml`
- documentation pages with example queries
- academic papers containing SPARQL or competency questions (CQs)

### Process (deterministic, Python)

- Clone repositories.
- Extract:
  - `.rq` and `.sparql` files
  - embedded SPARQL in code or documentation
- Normalise whitespace and prefixes.
- Deduplicate by hash.
- Record provenance:
  - repository URL
  - file path
  - commit hash
  - line spans (if available)

### Output

`kg_queries.jsonl` (query records with raw SPARQL, clean SPARQL, and evidence)

No filtering and no LLM use at this stage.

---

## 3.5. Evidence enrichment (`kg_queries.jsonl`)

**Goal:** enrich query records with human-readable evidence from sources.

### Inputs

- `kg_queries.jsonl`
- repositories listed in `seeds.yaml`
- documentation pages (optional)
- academic papers (optional)

### Process (deterministic)

- For repo files:
  - Extract leading comment blocks in `.rq`/`.sparql` files.
  - For Markdown files, pair each fenced `sparql` block with the nearest
    preceding paragraph as a description.
- Store extracted text as `evidence` items linked to the query record.

### Output

`kg_queries.jsonl` (updated in-place with evidence items)

---

## 4. Query execution and filtering (`run_queries.jsonl`)

**Goal:** keep only queries that actually run.

### Inputs

- `kg_queries.jsonl`
- SPARQL endpoints from `seeds.yaml`

### Process (deterministic)

For each query:

- Execute against the endpoint with a timeout.
- Record:
  - execution status (`ok`, `empty`, `timeout`, `parse_error`, `auth`, etc.)
  - timestamp
  - optional first result row
- Store latest run and latest successful run in the same record.

### Output

`run_queries.jsonl` (query records with run metadata)

This step establishes **ground-truth executability** for each query record.

---

## 5. Natural-language question and confidence generation (`pairs.jsonl`)

**Goal:** create human-readable NL–SPARQL pairs with confidence estimates.

### Inputs

- `run_queries.jsonl`
- KG descriptions from `kgs.jsonl`
- optional sample result rows

### Process (LLM with schema enforcement)

For each runnable query, generate an object of the form (stored in `llm_output`):

    {
      "ranked_evidence_phrases": [
        {
          "text": "...",
          "evidence_id": "e12",
          "source_type": "query_comment",
          "rank": 1,
          "verbatim": true
        }
      ],
      "nl_question": "...",
      "nl_question_origin": {
        "mode": "verbatim|paraphrased|generated",
        "evidence_ids": ["e12", "e7"],
        "primary_evidence_id": "e12"
      },
      "confidence": 92,
      "confidence_rationale": "..."
    }

Guidelines:

- Prefer **simple, human phrasing**.
- Avoid ontology jargon unless unavoidable.
- Lower confidence if semantics are ambiguous.

### Evidence prioritization for LLM input

Provide the full evidence list to the LLM and specify a preference order by type:

1. `query_comment` (SPARQL comments)
2. `doc_query_desc` / `web_query_desc` / `readme_query_desc`
3. `cq_item`
4. general KG descriptions (`kg_summary`, `doc_summary`, `readme_summary`, `web_summary`, `repo_summary`)

Optionally run a second **consistency-check pass** to downgrade overconfident pairs.

### Filtering rule

- Keep only pairs with `confidence ≥ 85`.
- Lower-confidence pairs go to review or discard.

### Output

`pairs.jsonl`, one pair per line, including:

- natural-language question
- SPARQL query
- confidence score and rationale
- provenance
- endpoint
- test status

This file is the **core dataset**.

---

## 6. Academic paper integration (parallel track)

For each KG:

- Identify canonical papers.
- Extract:
  - SPARQL examples
  - competency questions (CQs)

If only CQs exist:

- Optionally draft SPARQL (marked as `crafted_from_cq`).
- Assign lower confidence unless verified against an endpoint.

Paper-derived queries pass through the **same pipeline** as repo-derived ones.

---

## 7. Outputs and intended use

At minimum, the project produces:

- `seeds.yaml` – control input
- `kgs.jsonl` – KG catalogue (Quagga-ready)
- `pairs.jsonl` – validated NL–SPARQL pairs

These outputs can be:

- ingested into Quagga
- used for evaluation or benchmarking
- published as a dataset
- extended with additional KGs

---

## 8. Why this workflow works

- Every artefact is reproducible.
- Every query is runnable or explicitly marked otherwise.
- Every NL question has an explicit confidence estimate.
- Provenance is preserved end-to-end.
- LLM use is restricted to tasks where it adds value (language, summarisation).

---

## 9. Current status

- Git repository initialised and pushed
- `seeds.yaml` committed (initial Polifonia KGs)
- SSH authentication confirmed

**Next step:** implement `build_kgs.py` or run one KG (e.g. MEETUPS) end-to-end as a reference implementation.
