from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
import mlflow

from api_client import PromptBenchmarkApiClient
from metrics import summarize_results
from schemas import BenchmarkCase, BenchmarkConfig, ClueBenchmarkRecord, RoundBenchmarkRecord

mlflow.set_tracking_uri("sqlite:///experiments/prompt_benchmarking/mlflow.db")
mlflow.set_experiment("prompt_benchmarking")

SUPPORTED_TECHNIQUES = {"zero_shot", "few_shot", "reasoning_guided", "meta"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark clue-generation prompt techniques.")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument(
        "--cases",
        default="experiments/prompt_benchmarking/benchmark_cases.json",
    )
    parser.add_argument(
        "--technique",
        default="few_shot",
        choices=[*sorted(SUPPORTED_TECHNIQUES), "all"],
        help="Prompt technique to benchmark, or 'all' to run every supported technique.",
    )
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
    client = PromptBenchmarkApiClient(
        backend_url=args.backend_url,
        timeout_seconds=args.timeout_seconds,
    )

    techniques = (
        sorted(SUPPORTED_TECHNIQUES)
        if args.technique == "all"
        else [args.technique]
    )
    for technique in techniques:
        config = BenchmarkConfig(
            backend_url=args.backend_url,
            technique=technique,
            repetitions=args.repetitions,
            timeout_seconds=args.timeout_seconds,
        )
        await _run_technique_benchmark(
            client=client,
            config=config,
            cases=cases,
            output_dir=output_dir,
            timestamp=timestamp,
        )


async def _run_technique_benchmark(
    client: PromptBenchmarkApiClient,
    config: BenchmarkConfig,
    cases: list[BenchmarkCase],
    output_dir: Path,
    timestamp: str,
) -> None:
    print(
        f"Running {config.technique} against {config.backend_url}. "
        "The prompt technique will be sent with each benchmark round request."
    )
    round_records: list[RoundBenchmarkRecord] = []
    clue_records: list[ClueBenchmarkRecord] = []

    for _ in range(config.repetitions):
        for case in cases:
            round_record, case_clues = await client.run_case(case, config.technique)
            round_records.append(round_record)
            clue_records.extend(case_clues)

    summary = summarize_results(config.technique, round_records, clue_records)
    rounds_path = output_dir / f"rounds_{config.technique}_{timestamp}.jsonl"
    clues_path = output_dir / f"clues_{config.technique}_{timestamp}.jsonl"
    summary_path = output_dir / f"summary_{config.technique}_{timestamp}.json"
    _write_jsonl(
        rounds_path,
        [record.model_dump() for record in round_records],
    )
    _write_jsonl(
        clues_path,
        [record.model_dump() for record in clue_records],
    )
    summary_path.write_text(
        json.dumps(summary.model_dump(), indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary.model_dump(), indent=2))

    # MLflow logging
    with mlflow.start_run(run_name=f"benchmark_{config.technique}_{timestamp}"):
        mlflow.log_param("prompting technique", config.technique)
        mlflow.log_param("repetitions", config.repetitions)
        mlflow.log_param("case count", len(cases))

        mlflow.log_metric("average_latency_ms", summary.average_latency_ms)
        mlflow.log_metric("round_success_rate", summary.round_success_rate)
        mlflow.log_metric("generation_failed_rate", summary.generation_failed_rate)
        mlflow.log_metric("duplicate_clue_rate", summary.duplicate_clue_rate)
        mlflow.log_metric("secret_word_leak_rate", summary.secret_word_leak_rate)
        mlflow.log_metric("empty_clue_rate", summary.empty_clue_rate)
        mlflow.log_metric("average_clue_word_count", summary.average_clue_word_count)

        mlflow.log_artifact(str(rounds_path))
        mlflow.log_artifact(str(clues_path))
        mlflow.log_artifact(str(summary_path))


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
