from __future__ import annotations

from collections import Counter, defaultdict

from schemas import BenchmarkSummary, ClueBenchmarkRecord, RoundBenchmarkRecord


def summarize_results(
    technique: str,
    rounds: list[RoundBenchmarkRecord],
    clues: list[ClueBenchmarkRecord],
) -> BenchmarkSummary:
    return BenchmarkSummary(
        technique=technique,
        average_latency_ms=_average([round_record.latency_ms for round_record in rounds]),
        round_success_rate=_rate([round_record.success for round_record in rounds]),
        generation_failed_rate=_rate(
            [round_record.status == "generation_failed" for round_record in rounds]
        ),
        duplicate_clue_rate=_duplicate_clue_rate(clues),
        secret_word_leak_rate=_secret_word_leak_rate(clues),
        empty_clue_rate=_rate([not clue.clue.strip() for clue in clues]),
        average_clue_word_count=_average([len(clue.clue.split()) for clue in clues]),
    )


def _average(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for flag in flags if flag) / len(flags))


def _duplicate_clue_rate(clues: list[ClueBenchmarkRecord]) -> float:
    if not clues:
        return 0.0

    clues_by_round: dict[str, list[str]] = defaultdict(list)
    for clue in clues:
        normalized = _normalize_text(clue.clue)
        if normalized:
            clues_by_round[clue.round_id].append(normalized)

    duplicate_flags: list[bool] = []
    for round_clues in clues_by_round.values():
        counts = Counter(round_clues)
        duplicate_flags.extend(count > 1 for count in counts.values())

    return _rate(duplicate_flags)


def _secret_word_leak_rate(clues: list[ClueBenchmarkRecord]) -> float:
    if not clues:
        return 0.0

    return _rate(
        [
            _normalize_text(clue.secret_word) in _normalize_text(clue.clue)
            for clue in clues
        ]
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())
