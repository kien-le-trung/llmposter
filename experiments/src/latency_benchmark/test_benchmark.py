import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import httpx

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backend_handler.api_client import BackendRequestArtifact  # noqa: E402
from latency_benchmark.report import stats, summarize  # noqa: E402
from latency_benchmark.stub import create_app  # noqa: E402
from runner.run_latency import LatencyClient, cells, parse_args  # noqa: E402


def test_default_matrices_are_bounded_and_live_is_explicit():
    controlled = parse_args([])
    assert controlled.mode == "controlled"
    assert controlled.repetitions == 5
    live = parse_args(["--mode", "live"])
    assert live.discussion_seconds == 90
    assert sum(cell.parallel for cell in cells(live)) == 15
    assert len(cells(controlled)) == 24


def test_ai_only_client_preserves_partial_results_and_observations(monkeypatch):
    async def run(fail_vote):
        api = LatencyClient("http://stub", poll_interval_seconds=0)
        rounds, observations, calls = [], [], []
        now = datetime.now(UTC).isoformat()
        statuses = iter(["generating_clues", "discussion", "ready_to_vote"])

        async def request(client, method, path, json=None):
            calls.append((method, path, json))
            if path.endswith("/vote"):
                if fail_vote:
                    raise TimeoutError()
                payload = {"agent_votes": []}
            else:
                payload = {"id": "round", "status": next(statuses), "created_at": now,
                           "playing_order": [{"id": "a"}], "discussion": {"messages": [
                               {"id": "message", "sequence": 1, "created_at": now,
                                "text": "private generated text"},
                           ]}}
            return BackendRequestArtifact(method, path, json, 200, payload, None, 1, now), payload

        monkeypatch.setattr(api, "_request_json", request)
        result = await api.play(None, "cell", False, 1, rounds, observations)
        assert calls[0][2]["include_human"] is False
        assert calls[-1] == ("POST", "/rounds/round/vote", {})
        assert result["outcome"] == ("failure" if fail_vote else "complete")
        assert len(observations) == 1
        assert "private" not in json.dumps([rounds, observations])
        if fail_vote:
            assert result["failed_phase"] == "voting"
            assert result["round_id"] == "round"

    asyncio.run(run(False))
    asyncio.run(run(True))


def test_report_excludes_warmup_and_preserves_failed_games(tmp_path):
    rounds = [{"run_id": "cell", "round_id": "warm", "warmup": True,
               "latency_ms": 9999, "outcome": "complete"},
              {"run_id": "cell", "round_id": "round", "warmup": False,
               "latency_ms": 20, "outcome": "failure"}]
    events = [{"event": "provider_http", "round_id": rid, "run_id": "cell",
               "duration_ms": elapsed, "start_ns": 1, "end_ns": 2, "outcome": "ok"}
              for rid, elapsed in (("warm", 9999), ("round", 10))]
    result = summarize(tmp_path, rounds, events, [])
    assert result["measured_games"] == 1
    assert result["failed_games"] == 1
    provider = next(row for row in result["metrics"] if row["metric"] == "provider_http")
    assert provider["max"] == 10
    assert provider["small_sample"]
    assert (tmp_path / "report.html").exists()
    assert stats([])["count"] == 0
    assert stats([1, 2, 3, 4])["p95"] == 4
    assert any(row["metric"] == "game_elapsed_until_failure" for row in result["metrics"])


def test_empty_partial_run_still_has_a_report(tmp_path):
    result = summarize(tmp_path, [], [], [], runner_error="TimeoutError")
    assert result["runner_error_type"] == "TimeoutError"
    assert "partial" in (tmp_path / "report.html").read_text()
    assert (tmp_path / "turns.csv").exists()


def test_delayed_http_stub_returns_structured_output_and_faults():
    async def run():
        app = create_app(delay=.01, scenario="rate_limit")
        transport = httpx.ASGITransport(app=app)
        payload = {"messages": [{"content": 'JSON schema:\n{"properties":{"clue":{}}}'}]}
        async with httpx.AsyncClient(transport=transport, base_url="http://stub") as client:
            assert (await client.post("/v1/chat/completions", json=payload)).status_code == 429
            responses = await asyncio.gather(*(client.post("/v1/chat/completions", json=payload)
                                               for _ in range(4)))
            assert app.state.peak == 4
            for response in responses:
                assert json.loads(response.json()["choices"][0]["message"]["content"])["clue"]

    asyncio.run(run())


def test_opening_comparison_retains_missing_speakers(tmp_path):
    from latency_benchmark.comparison import comparison_report
    games = []
    events = []
    for capacity, end in ((1, 800), (4, 200)):
        run = f"0{capacity}-discussion-cap{capacity}-rounds1-client4-delay0.2"
        rid = str(capacity)
        games.append({"run_id": run, "round_id": rid, "playing_order": "a,b", "batch_index": 0})
        events.append({"event": "discussion_opened", "round_id": rid, "mono_ns": 0})
        for agent in ("a", "b"):
            events.append({"event": "message_published", "round_id": rid, "agent_id": agent, "mono_ns": end*1000000})
    result = comparison_report(tmp_path, games, list(reversed(events)))
    assert result[0]["speedup"] == 4
    assert result[0]["reduction_percent"] == 75
    events.pop()
    assert comparison_report(tmp_path, games, events) == []
