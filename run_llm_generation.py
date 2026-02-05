#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_examples(path: Optional[Path], limit: int = 2) -> str:
    if path is None or not path.exists():
        return ""
    examples = load_jsonl(path)[:limit]
    if not examples:
        return ""
    return json.dumps(examples, ensure_ascii=False, indent=2)


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: brace scan for first valid object.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
    return None


def validate_output(obj: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    try:
        import jsonschema  # type: ignore
    except Exception:
        required = schema.get("required", [])
        missing = [k for k in required if k not in obj]
        if missing:
            return False, f"Missing required keys: {missing}"
        return True, None

    try:
        jsonschema.validate(instance=obj, schema=schema)
        return True, None
    except Exception as e:
        return False, str(e)


def build_system_prompt(
    base_prompt: str,
    schema: Dict[str, Any],
    examples_text: str,
) -> str:
    parts = [base_prompt.strip(), "\nOutput schema (JSON):", json.dumps(schema, ensure_ascii=False, indent=2)]
    if examples_text:
        parts.extend(["\nFew-shot examples:", examples_text])
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NL generation with OpenAI over LLM inputs JSONL.")
    parser.add_argument("--input", default="prompts/llm_nl_generation.test_inputs.jsonl")
    parser.add_argument("--prompt", default="prompts/llm_nl_generation.prompt.txt")
    parser.add_argument("--schema", default="schemas/llm_output.schema.json")
    parser.add_argument("--examples", default="prompts/llm_nl_generation.examples.jsonl")
    parser.add_argument("--output", default="llm_outputs.jsonl")
    parser.add_argument("--errors", default="llm_outputs.errors.jsonl")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--max-records", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    input_path = Path(args.input)
    prompt_path = Path(args.prompt)
    schema_path = Path(args.schema)
    examples_path = Path(args.examples) if args.examples else None
    out_path = Path(args.output)
    err_path = Path(args.errors)

    inputs = load_jsonl(input_path)
    if args.max_records > 0:
        inputs = inputs[: args.max_records]
    base_prompt = prompt_path.read_text(encoding="utf-8")
    schema = load_json(schema_path)
    examples_text = load_examples(examples_path)
    system_prompt = build_system_prompt(base_prompt, schema, examples_text)

    client = OpenAI()
    ok_count = 0
    err_count = 0

    with out_path.open("w", encoding="utf-8") as out_f, err_path.open("w", encoding="utf-8") as err_f:
        for idx, payload in enumerate(inputs, start=1):
            user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
            try:
                resp = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = (resp.output_text or "").strip()
                parsed = extract_first_json_object(text)
                if parsed is None:
                    raise ValueError("No JSON object found in model output")
                valid, validation_error = validate_output(parsed, schema)
                if not valid:
                    raise ValueError(f"Schema validation failed: {validation_error}")
                out_rec = {
                    "query_id": payload.get("query_id"),
                    "query_label": payload.get("query_label"),
                    "kg_id": payload.get("kg_id"),
                    "llm_output": parsed,
                    "model": args.model,
                }
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                ok_count += 1
                print(f"[{idx}/{len(inputs)}] ok {payload.get('query_label')}")
            except Exception as e:
                err_rec = {
                    "query_id": payload.get("query_id"),
                    "query_label": payload.get("query_label"),
                    "kg_id": payload.get("kg_id"),
                    "error": str(e),
                }
                err_f.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                err_count += 1
                print(f"[{idx}/{len(inputs)}] error {payload.get('query_label')}: {e}")

    print(f"Wrote {ok_count} outputs to {out_path.resolve()}")
    if err_count:
        print(f"Wrote {err_count} errors to {err_path.resolve()}")


if __name__ == "__main__":
    main()
