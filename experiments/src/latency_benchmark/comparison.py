"""Opening-message comparisons; incomplete rounds remain explicit."""
import json
import re
import statistics
from collections import defaultdict


def comparison_report(output, rounds, events):
    rows = []
    batches = defaultdict(list)
    for game in rounds:
        if game.get("warmup"):
            continue
        ev = [e for e in events if e.get("round_id") == game["round_id"]]
        opened = next((e for e in ev if e["event"] == "discussion_opened"), None)
        expected = set(game.get("playing_order", "").split(",")) - {"human", ""}
        first = {}
        turns = {}
        for e in sorted(ev, key=lambda e: e.get("mono_ns", 0)):
            agent = e.get("agent_id")
            if agent not in expected:
                continue
            if e["event"] == "message_published":
                first.setdefault(agent, e["mono_ns"])
            if e["event"] == "discussion_turn" and e.get("outcome") in {"published", "silence"}:
                turns.setdefault(agent, e["mono_ns"])
        def elapsed(values):
            return (max(values.values()) - opened["mono_ns"]) / 1e6 if opened and expected and expected <= values.keys() else None
        row = {"run_id": game["run_id"], "round_id": game["round_id"], "batch_index": game.get("batch_index", 0),
               "all_agents_opening_ms": elapsed(first), "initial_turn_completion_ms": elapsed(turns),
               "first_message_ms": (min(first.values())-opened["mono_ns"])/1e6 if first and opened else None}
        rows.append(row)
        batches[(game["run_id"], row["batch_index"])].append(row)
    grouped = defaultdict(list)
    for (run, batch), values in batches.items():
        complete = [r["all_agents_opening_ms"] for r in values]
        if all(v is not None for v in complete):
            grouped[run].append(statistics.mean(complete))
    comparisons = []
    for run, values in grouped.items():
        if "-cap1-" not in run:
            continue
        suffix = run.split("-cap1-", 1)[1]
        other = next((r for r in grouped if "-cap4-" in r and r.split("-cap4-", 1)[1] == suffix), None)
        if other:
            serial, concurrent = statistics.median(values), statistics.median(grouped[other])
            comparisons.append({"serial": run, "concurrent": other, "serial_batches": len(values),
                                "concurrent_batches": len(grouped[other]), "serial_ms": serial, "concurrent_ms": concurrent,
                                "speedup": serial/concurrent if concurrent else None,
                                "reduction_percent": 100*(1-concurrent/serial) if serial else None})
    (output/"opening_measurements.json").write_text(json.dumps(rows, indent=2))
    (output/"comparisons.json").write_text(json.dumps(comparisons, indent=2))
    text = "# Concurrency latency pilot\n\nDescriptive results; incomplete openings are excluded from latency comparisons and retained in opening_measurements.json.\n\n"
    for c in comparisons:
        text += f"- {c['serial']} versus {c['concurrent']}: {c['serial_ms']:.1f} ms versus {c['concurrent_ms']:.1f} ms; {c['speedup']:.2f}x speedup, {c['reduction_percent']:.1f}% reduction.\n"
    text += "\nBoth conditions use asyncio. This measures request overlap, not a runtime comparison. Fixed discussion timers mask total-game speedup. SQLite, provider routing, silence, retries, and five-batch sampling limit generalization.\n"
    (output/"findings.md").write_text(text)
    return comparisons
