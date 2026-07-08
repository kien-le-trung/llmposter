from __future__ import annotations

from schemas import BenchmarkSummary, RoundLatencyRecord


def summarize_results(
    mode: str,
    prompt_technique: str,
    repetitions: int,
    case_count: int,
    records: list[RoundLatencyRecord],
) -> BenchmarkSummary:
    successful_records = [record for record in records if record.success]
    latencies = sorted(record.total_latency_ms for record in successful_records)
    create_latencies = [
        record.create_latency_ms for record in successful_records if record.create_latency_ms is not None
    ]
    continuation_latencies = [
        record.continuation_latency_ms
        for record in successful_records
        if record.continuation_latency_ms is not None
    ]

    return BenchmarkSummary(
        mode=mode,
        prompt_technique=prompt_technique,
        repetitions=repetitions,
        case_count=case_count,
        round_count=len(records),
        success_rate=_rate(len(successful_records), len(records)),
        generation_failed_rate=_rate(
            sum(1 for record in records if record.status == "generation_failed"),
            len(records),
        ),
        timeout_rate=_rate(
            sum(1 for record in records if record.status == "timeout"),
            len(records),
        ),
        average_latency_ms=_mean(latencies),
        median_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        average_create_latency_ms=_mean(create_latencies),
        average_continuation_latency_ms=_mean(continuation_latencies),
        average_generated_agent_clue_count=_mean(
            [record.generated_agent_clue_count for record in successful_records]
        ),
    )


def _mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0

    index = round((len(values) - 1) * (percentile / 100))
    return float(values[index])
