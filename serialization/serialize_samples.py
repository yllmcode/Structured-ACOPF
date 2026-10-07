

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from .dataset import enumerate_records
from .matpower import load_matpower_case
from .proposed import serialize_sample


SYSTEM_PROMPT = (
    "You are an AC optimal power flow surrogate. Infer the complete optimum from "
    "the supplied grid record. Output only the requested serialization."
)


def encoder_prompt(tokenizer, instruction: str, input_text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction + "\n\n" + input_text},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create proposed ACOPF JSONL samples")
    parser.add_argument("--case", required=True)
    parser.add_argument("--data", required=True, help="NPZ file or sharded NPZ directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--tokenizer", help="Llama tokenizer id or local path")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    case_path = Path(args.case)
    case = load_matpower_case(case_path)
    records = enumerate_records(args.data)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("no source records found")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prompt_tokens: list[int] = []
    input_characters: list[int] = []
    opened_path = None
    shard = None
    arrays = None
    try:
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for sequence_index, (path, row) in enumerate(records):
                if path != opened_path:
                    if shard is not None:
                        shard.close()
                    opened_path = path
                    shard = np.load(path)
                    arrays = {key: shard[key] for key in shard.files}
                assert arrays is not None
                source_index = (
                    int(arrays["source_index"][row])
                    if "source_index" in arrays
                    else sequence_index
                )
                sample = serialize_sample(
                    case, arrays, row, source_index=source_index
                )
                handle.write(
                    json.dumps(
                        {
                            "instruction": sample.instruction,
                            "input": sample.input_text,
                            "output": sample.output_text,
                            "method": "proposed",
                            "case": sample.case,
                            "source_index": sample.source_index,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                input_characters.append(len(sample.input_text))
                if tokenizer is not None:
                    prompt = encoder_prompt(
                        tokenizer, sample.instruction, sample.input_text
                    )
                    prompt_tokens.append(
                        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                    )
    finally:
        if shard is not None:
            shard.close()

    summary = {
        "schema_version": 1,
        "case": case.name,
        "method": "proposed",
        "samples": len(records),
        "static_case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "jsonl_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "input_characters": {
            "mean": statistics.fmean(input_characters),
            "minimum": min(input_characters),
            "maximum": max(input_characters),
        },
        "prompt_tokens": (
            {
                "tokenizer": Path(args.tokenizer).name,
                "scope": "system/user chat template plus instruction and serialized input; target excluded",
                "mean": statistics.fmean(prompt_tokens),
                "minimum": min(prompt_tokens),
                "maximum": max(prompt_tokens),
                "rounded_mean": round(statistics.fmean(prompt_tokens)),
            }
            if prompt_tokens
            else None
        ),
    }
    summary_path = Path(args.summary) if args.summary else destination.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
