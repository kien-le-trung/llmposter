from metrics import summarize_results
from schemas import ClueBenchmarkRecord, RoundBenchmarkRecord


def make_round(status: str, latency_ms: float, success: bool) -> RoundBenchmarkRecord:
    return RoundBenchmarkRecord(
        technique="few_shot",
        round_id="round-1",
        secret_word="music",
        imposter_hint="sound",
        status=status,
        latency_ms=latency_ms,
        success=success,
    )


def make_clue(round_id: str, secret_word: str, clue: str) -> ClueBenchmarkRecord:
    return ClueBenchmarkRecord(
        technique="few_shot",
        round_id=round_id,
        secret_word=secret_word,
        imposter_hint="sound",
        agent_id="agent_a",
        agent_name="Agent A",
        clue=clue,
        inference_mode="fake",
    )


def test_average_latency() -> None:
    summary = summarize_results(
        "few_shot",
        [make_round("ready_to_vote", 100.0, True), make_round("ready_to_vote", 300.0, True)],
        [],
    )
    assert summary.average_latency_ms == 200.0
    assert summary.round_success_rate == 1.0


def test_status_and_empty_text_rates() -> None:
    summary = summarize_results(
        "few_shot",
        [make_round("generation_failed", 100.0, False)],
        [make_clue("round-1", "music", "")],
    )
    assert summary.generation_failed_rate == 1.0
    assert summary.empty_clue_rate == 1.0


def test_word_reuse_metric() -> None:
    summary = summarize_results(
        "few_shot",
        [make_round("ready_to_vote", 100.0, True)],
        [make_clue("round-1", "music", "music hall")],
    )
    assert summary.secret_word_leak_rate == 1.0


def test_duplicate_metric() -> None:
    summary = summarize_results(
        "few_shot",
        [make_round("ready_to_vote", 100.0, True)],
        [
            make_clue("round-1", "music", "rhythm crowd"),
            make_clue("round-1", "music", "rhythm crowd"),
        ],
    )
    assert summary.duplicate_clue_rate == 1.0


def test_empty_inputs() -> None:
    summary = summarize_results("few_shot", [], [])
    assert summary.average_latency_ms == 0.0
    assert summary.round_success_rate == 0.0
