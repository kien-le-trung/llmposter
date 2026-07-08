from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from api_client import BatchingBenchmarkApiClient
from metrics import summarize_results
from schemas import BenchmarkCase, BenchmarkConfig


SUPPORTED_MODES = {"batched", "unbatched"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batched vs per-agent clue prompting.")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument(
        "--cases",
        default="experiments/batching_benchmarking/benchmark_cases.json",
    )
    parser.add_argument("--mode", required=True, choices=sorted(SUPPORTED_MODES))
    parser.add_argument("--prompt-technique", default="few_shot")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="experiments/batching_benchmarking/results",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N completed cases. Use 0 to disable progress output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print start and end details for every benchmark case.",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Print failed case error messages as they happen.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cases = _load_cases(Path(args.cases))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        backend_url=args.backend_url,
        mode=args.mode,
        prompt_technique=args.prompt_technique,
        repetitions=args.repetitions,
        timeout_seconds=args.timeout_seconds,
    )
    client = BatchingBenchmarkApiClient(
        backend_url=config.backend_url,
        timeout_seconds=config.timeout_seconds,
    )
    records = []
    total_cases = config.repetitions * len(cases)
    completed_cases = 0
    for repetition_index in range(1, config.repetitions + 1):
        for case_index, case in enumerate(cases, start=1):
            if args.verbose:
                print(
                    f"[start] repetition={repetition_index}/{config.repetitions} "
                    f"case={case_index}/{len(cases)} word={case.secret_word!r} "
                    f"mode={config.mode}",
                    flush=True,
                )
            record = await client.run_case(
                case,
                config.mode,
                config.prompt_technique,
            )
            records.append(record)
            completed_cases += 1

            if args.verbose or (
                args.progress_every > 0 and completed_cases % args.progress_every == 0
            ):
                print(
                    f"[done] {completed_cases}/{total_cases} "
                    f"repetition={repetition_index}/{config.repetitions} "
                    f"case={case_index}/{len(cases)} word={case.secret_word!r} "
                    f"success={record.success} status={record.status!r} "
                    f"total_ms={record.total_latency_ms:.1f} "
                    f"create_ms={record.create_latency_ms:.1f} "
                    f"continuation_ms={record.continuation_latency_ms:.1f} "
                    f"polls={record.poll_count}",
                    flush=True,
                )
            if args.show_errors and record.error:
                print(
                    f"[error] repetition={repetition_index} case={case_index} "
                    f"word={case.secret_word!r}: {record.error}",
                    flush=True,
                )

    summary = summarize_results(
        mode=config.mode,
        prompt_technique=config.prompt_technique,
        repetitions=config.repetitions,
        case_count=len(cases),
        records=records,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    records_path = output_dir / f"rounds_{config.mode}_{timestamp}.jsonl"
    summary_path = output_dir / f"summary_{config.mode}_{timestamp}.json"
    records_path.write_text(
        "".join(json.dumps(record.model_dump(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary.model_dump(), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary.model_dump(), indent=2))


def _load_cases(path: Path) -> list[BenchmarkCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkCase.model_validate(case) for case in raw_cases]


if __name__ == "__main__":
    asyncio.run(main())
