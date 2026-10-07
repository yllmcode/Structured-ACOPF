

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .matpower import load_matpower_case
from .proposed import parse_target


ROOT = Path(__file__).resolve().parents[1]
CASES = ("case9", "case30", "case118")
SPLITS = (("train", 9000), ("test", 1000))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def main() -> None:
    report: dict[str, object] = {}
    for name in CASES:
        case_path = ROOT / "cases" / f"{name}.m"
        sample_dir = ROOT / "samples" / name
        manifest_path = sample_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case = load_matpower_case(case_path)
        if manifest["case_sha256"] != sha256(case_path):
            raise SystemExit(f"{name}: original case hash mismatch")

        seen_indices: set[int] = set()
        case_report: dict[str, object] = {}
        for split, expected in SPLITS:
            jsonl_path = sample_dir / f"{split}.jsonl"
            summary_path = sample_dir / f"{split}.summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if manifest["splits"][split]["samples"] != expected:
                raise SystemExit(f"{name}/{split}: manifest sample count mismatch")
            if manifest["splits"][split]["jsonl_sha256"] != sha256(jsonl_path):
                raise SystemExit(f"{name}/{split}: manifest JSONL hash mismatch")
            if manifest["splits"][split]["summary_sha256"] != sha256(summary_path):
                raise SystemExit(f"{name}/{split}: manifest summary hash mismatch")
            if summary["samples"] != expected:
                raise SystemExit(f"{name}/{split}: summary sample count mismatch")
            if summary["static_case_sha256"] != sha256(case_path):
                raise SystemExit(f"{name}/{split}: summary case hash mismatch")
            if summary["jsonl_sha256"] != sha256(jsonl_path):
                raise SystemExit(f"{name}/{split}: summary JSONL hash mismatch")

            count = 0
            split_indices: set[int] = set()
            for line_number, record in iter_jsonl(jsonl_path):
                count += 1
                required = {
                    "instruction",
                    "input",
                    "output",
                    "method",
                    "case",
                    "source_index",
                }
                if set(record) != required:
                    raise SystemExit(
                        f"{name}/{split}:{line_number}: unexpected record fields"
                    )
                if record["method"] != "proposed" or record["case"] != name:
                    raise SystemExit(
                        f"{name}/{split}:{line_number}: record identity mismatch"
                    )
                source_index = int(record["source_index"])
                if source_index in split_indices or source_index in seen_indices:
                    raise SystemExit(
                        f"{name}/{split}:{line_number}: duplicate source index"
                    )
                split_indices.add(source_index)
                parsed = parse_target(record["output"], case)
                if (
                    len(parsed["pg_pu"]) != case.n_gen
                    or len(parsed["vm_pu"]) != case.n_bus
                    or len(parsed["pf_pu"]) != case.n_branch
                ):
                    raise SystemExit(
                        f"{name}/{split}:{line_number}: target dimensions mismatch"
                    )
            if count != expected:
                raise SystemExit(
                    f"{name}/{split}: expected {expected} records, found {count}"
                )
            seen_indices.update(split_indices)
            case_report[split] = {"samples": count, "jsonl_sha256": sha256(jsonl_path)}

        token_summary = json.loads(
            (sample_dir / "test.summary.json").read_text(encoding="utf-8")
        )["prompt_tokens"]
        if not token_summary:
            raise SystemExit(f"{name}: missing tokenizer evidence")
        case_report["rounded_mean_prompt_tokens"] = token_summary["rounded_mean"]
        report[name] = case_report

    if report["case118"]["rounded_mean_prompt_tokens"] != 3297:
        raise SystemExit("IEEE-118 token count does not match the revised documents")
    print(json.dumps({"status": "pass", "systems": report}, indent=2))


if __name__ == "__main__":
    main()
