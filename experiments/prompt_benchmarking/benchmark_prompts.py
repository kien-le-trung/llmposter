from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from api_client import PromptBenchmarkApiClient
from metrics import summarize_results
from schemas import BenchmarkCase, BenchmarkConfig, ClueBenchmarkRecord, RoundBenchmarkRecord

DEFAULT_TECHNIQUES = ("zero_shot", "few_shot", "reasoning_guided", "meta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark clue-generation prompt techniques.")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument(
        "--cases",
        default="experiments/prompt_benchmarking/benchmark_cases.json",
    )
    parser.add_argument("--techniques", nargs="+", default=list(DEFAULT_TECHNIQUES))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="experiments/prompt_benchmarking/results",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cases = _load_cases(Path(args.cases))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    all_round_records: list[RoundBenchmarkRecord] = []
    all_clue_records: list[ClueBenchmarkRecord] = []
    summaries = []

    for technique in args.techniques:
        config = BenchmarkConfig(
            backend_url=args.backend_url,
            technique=technique,
            repetitions=args.repetitions,
            timeout_seconds=args.timeout_seconds,
        )
        print(
            f"Running {config.technique} against {config.backend_url}. "
            "Ensure the backend was started with "
            f"CLUE_PROMPT_TECHNIQUE={config.technique}."
        )
        client = PromptBenchmarkApiClient(
            backend_url=config.backend_url,
            timeout_seconds=config.timeout_seconds,
        )

        round_records: list[RoundBenchmarkRecord] = []
        clue_records: list[ClueBenchmarkRecord] = []
        for _ in range(config.repetitions):
            for case in cases:
                round_record, case_clues = await client.run_case(case, config.technique)
                round_records.append(round_record)
                clue_records.extend(case_clues)

        all_round_records.extend(round_records)
        all_clue_records.extend(clue_records)
        summaries.append(summarize_results(config.technique, round_records, clue_records))

    _write_jsonl(
        output_dir / f"rounds_{timestamp}.jsonl",
        [record.model_dump() for record in all_round_records],
    )
    _write_jsonl(
        output_dir / f"clues_{timestamp}.jsonl",
        [record.model_dump() for record in all_clue_records],
    )
    (output_dir / f"summary_{timestamp}.json").write_text(
        json.dumps([summary.model_dump() for summary in summaries], indent=2),
        encoding="utf-8",
    )

    print(json.dumps([summary.model_dump() for summary in summaries], indent=2))


def _load_cases(path: Path) -> list[BenchmarkCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkCase.model_validate(case) for case in raw_cases]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
