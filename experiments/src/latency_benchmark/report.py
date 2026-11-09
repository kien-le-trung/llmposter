"""Portable CSV/JSON summaries and an offline interactive HTML report."""

from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_events(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A forcibly stopped child can leave one partial trailing event.
    return rows


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    result = {"count": len(ordered), "small_sample": len(ordered) < 20}
    if not ordered:
        return result
    for name, q in (("p50", .5), ("p90", .9), ("p95", .95), ("max", 1)):
        result[name] = ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output: Path, rounds: list[dict], events: list[dict], observations: list[dict],
              runner_error: str | None = None):
    warmup_ids = {row["round_id"] for row in rounds if row.get("warmup")}
    measured = [e for e in events if e.get("round_id") not in warmup_ids]
    by_round = defaultdict(list)
    for event in measured:
        by_round[event.get("round_id")].append(event)
    for row in rounds:
        milestones = {e["event"]: e for e in by_round[row["round_id"]]}
        for name, start, end in (
            ("clue_phase_ms", "round_created", "discussion_opened"),
            ("discussion_ms", "discussion_opened", "discussion_closed"),
        ):
            if start in milestones and end in milestones:
                row[name] = (milestones[end]["mono_ns"] - milestones[start]["mono_ns"]) / 1e6
    publications = {e.get("message_id"): e for e in measured if e["event"] == "message_published"}
    for row in observations:
        event = publications.get(row["message_id"])
        if event:
            row["published_at"] = event["at"]
            row["observation_delay_ms"] = (
                datetime.fromisoformat(row["observed_at"]) - datetime.fromisoformat(event["at"])
            ).total_seconds() * 1000

    turns = []
    turn_events = defaultdict(list)
    for event in measured:
        turn_events[(event.get("round_id"), event.get("turn_id"))].append(event)
    for event in measured:
        if event["event"] not in {"clue_graph", "discussion_turn", "agent_vote"}:
            continue
        row = {k: v for k, v in event.items() if k not in {"event_id", "at"}}
        related = turn_events[(event.get("round_id"), event.get("turn_id"))]
        eligible = next((e for e in related if e["event"] == "scheduler_eligible"), None)
        if eligible:
            row["scheduler_dispatch_ms"] = (event["start_ns"] - eligible["mono_ns"]) / 1e6
        for name in ("capacity_wait", "client_semaphore_wait", "provider_http",
                     "retry_backoff", "discussion_graph", "publication", "message_persistence"):
            row[name + "_ms"] = sum(e.get("duration_ms", 0) for e in related if e["event"] == name)
        if event["event"] == "clue_graph":
            persisted = next((e for e in related if e["event"] == "clue_persisted"), None)
            revealed = next((e for e in related if e["event"] == "clue_revealed"), None)
            if persisted:
                row["clue_persisted_ms"] = (persisted["mono_ns"] - event["start_ns"]) / 1e6
            if revealed:
                row["clue_revealed_ms"] = (revealed["mono_ns"] - event["start_ns"]) / 1e6
        if event["event"] == "discussion_turn" and event.get("outcome") == "published":
            pub = next((e for e in related if e["event"] == "message_published"), None)
            if pub:
                row["published_message_ms"] = (pub["mono_ns"] - event["start_ns"]) / 1e6
        turns.append(row)

    groups = defaultdict(list)
    outcomes = defaultdict(Counter)
    for event in measured:
        if "duration_ms" in event:
            groups[(event["run_id"], event["event"])].append(event["duration_ms"])
            if event["event"] in {"provider_http", "discussion_turn", "capacity_wait", "clue_graph", "agent_vote"}:
                label = f'{event["event"]}.outcome_{event.get("outcome", "unknown")}'
                groups[(event["run_id"], label)].append(event["duration_ms"])
        if event["event"] == "discussion_turn":
            outcomes[event["run_id"]][event.get("outcome", "unknown")] += 1
    for row in rounds:
        if not row.get("warmup"):
            name = "game_elapsed" if row["outcome"] == "complete" else "game_elapsed_until_failure"
            groups[(row["run_id"], name)].append(row["latency_ms"])
            for metric in ("clue_phase_ms", "discussion_ms", "voting_ms"):
                if metric in row:
                    groups[(row["run_id"], metric)].append(row[metric])
    metrics = [{"run_id": cell, "metric": metric, **stats(values)}
               for (cell, metric), values in sorted(groups.items())]
    rates = []
    for cell in sorted({e["run_id"] for e in measured}):
        cell_events = [e for e in measured if e["run_id"] == cell]
        opens = [e["mono_ns"] for e in cell_events if e["event"] == "discussion_opened"]
        closes = [e["mono_ns"] for e in cell_events if e["event"] == "discussion_closed"]
        intervals = []
        for rid in {e.get("round_id") for e in cell_events}:
            start = next((e["mono_ns"] for e in cell_events if e.get("round_id") == rid and e["event"] == "discussion_opened"), None)
            end = next((e["mono_ns"] for e in cell_events if e.get("round_id") == rid and e["event"] == "discussion_closed"), None)
            if start is not None and end is not None:
                intervals.append((start, end))
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        wall_seconds = sum(end-start for start, end in merged) / 1e9
        round_seconds = sum(end-start for start, end in intervals) / 1e9
        counts = outcomes[cell]
        total = sum(counts.values())
        rates.append({"run_id": cell, "discussion_window_seconds": wall_seconds,
                      "round_discussion_seconds": round_seconds,
                      "published_per_round_second": counts["published"] / round_seconds if round_seconds else 0,
                      "published_per_second": counts["published"] / wall_seconds if wall_seconds else 0,
                      "completed_turns_per_second": (counts["published"] + counts["silence"]) / wall_seconds
                      if wall_seconds else 0,
                      "outcomes": dict(counts),
                      "outcome_rates": {k: v / total for k, v in counts.items()} if total else {}})
    summary = {"metrics": metrics, "rates": rates,
               "measured_games": sum(not r.get("warmup", False) for r in rounds),
               "failed_games": sum(r["outcome"] != "complete" for r in rounds if not r.get("warmup")),
               "notes": ["Percentiles use nearest rank; fewer than 20 samples are descriptive only.",
                         "HTTP duration includes network and provider-side queueing, not just model compute.",
                         "Scheduler wait includes intentional debounce, cooldown and silence timers.",
                         "Cancelled/expired durations end at cancellation; use outcome-specific rows for completed latency.",
                         "Game duration includes the fixed discussion timer; compare phases and throughput."]}
    if runner_error:
        summary["runner_error_type"] = runner_error
        summary["notes"].append(f"Runner stopped early ({runner_error}); results are partial.")
    required = {"round_created", "discussion_opened", "discussion_closed", "round_complete"}
    summary["games_missing_lifecycle_events"] = [
        row["round_id"] for row in rounds if row["outcome"] == "complete" and not row.get("warmup")
        and not required.issubset({e["event"] for e in by_round[row["round_id"]]})
    ]
    if summary["games_missing_lifecycle_events"]:
        summary["notes"].append("Some completed games have incomplete telemetry; inspect metrics.json.")
    from latency_benchmark.comparison import comparison_report
    summary["comparisons"] = comparison_report(output, rounds, measured)
    write_csv(output / "rounds.csv", rounds)
    write_csv(output / "turns.csv", turns)
    write_csv(output / "messages.csv", observations)
    write_json(output / "metrics.json", summary)
    summary["notes"].append("Opening comparisons are in comparisons.json and findings.md; incomplete openings remain explicit in opening_measurements.json.")
    timeline = [{k: e.get(k) for k in ("run_id", "phase", "agent_id", "round_id", "start_ns",
                                      "end_ns", "duration_ms", "outcome")}
                for e in measured if e["event"] == "provider_http"]
    data = json.dumps({**summary, "timeline": timeline}).replace("<", "\\u003c")
    (output / "report.html").write_text(HTML.replace("__DATA__", data), encoding="utf-8")
    return summary


HTML = """<!doctype html><html><meta charset="utf-8"><title>AI game latency</title>
<style>body{font:15px system-ui;max-width:1200px;margin:30px auto;padding:15px;background:#f6f8fb}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}
svg{background:white;width:100%;border:1px solid #ddd}select{padding:8px}small{color:#555}</style>
<h1>AI-only game latency</h1><p id="summary"></p><select id="cell"></select>
<h2>Opening comparison</h2><pre id="comparisons"></pre><h2>Latency distributions</h2><p>p50 / p90 / p95 / max in milliseconds. * Fewer than 20 samples.</p>
<table><thead><tr><th>Metric</th><th>Samples</th><th>p50</th><th>p90</th><th>p95</th><th>Max</th></tr></thead><tbody id="metrics"></tbody></table>
<h2>Queue wait and HTTP duration</h2><p>Bars compare p95; HTTP includes upstream waiting.</p><svg id="bars" viewBox="0 0 1100 180"></svg>
<h2>Request overlap</h2><p>One row per provider HTTP attempt; hover for identity and duration. At most 400 rows shown per cell.</p><svg id="timeline"></svg>
<h2>Throughput and turn outcomes</h2><pre id="rates"></pre><ul id="notes"></ul>
<script>const data=__DATA__;const ns='http://www.w3.org/2000/svg';
const [cell,summary,notes,metrics,rates,bars,timeline]=['cell','summary','notes','metrics','rates','bars','timeline'].map(id=>document.getElementById(id));
function node(tag,attrs,text){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;return e}
const cells=[...new Set(data.metrics.map(x=>x.run_id))];for(const c of cells){let o=document.createElement('option');o.value=c;o.textContent=c;cell.append(o)}
summary.textContent=`${data.measured_games} measured games; ${data.failed_games} failed games. Warm-up excluded.`;
for(const n of data.notes){let li=document.createElement('li');li.textContent=n;notes.append(li)}
function draw(){const c=cell.value;metrics.replaceChildren();const ms=data.metrics.filter(x=>x.run_id===c);
for(const m of ms){let tr=document.createElement('tr');for(const v of [m.metric,m.count+(m.small_sample?' *':''),m.p50,m.p90,m.p95,m.max]){let td=document.createElement('td');td.textContent=typeof v==='number'?v.toFixed(1):v??'';tr.append(td)}metrics.append(tr)}
document.getElementById('comparisons').textContent=JSON.stringify((data.comparisons||[]).filter(x=>x.serial===c||x.concurrent===c),null,2);
rates.textContent=JSON.stringify(data.rates.find(x=>x.run_id===c)||{},null,2);
bars.replaceChildren();const bs=ms.filter(x=>['capacity_wait','client_semaphore_wait','provider_http.outcome_ok'].includes(x.metric));const max=Math.max(1,...bs.map(x=>x.p95));bs.forEach((x,i)=>{bars.append(node('text',{x:10,y:i*50+25},x.metric));bars.append(node('rect',{x:230,y:i*50+8,width:780*x.p95/max,height:25,fill:'#4576cc'}));bars.append(node('text',{x:235,y:i*50+26,fill:'white'},x.p95.toFixed(1)+' ms'))});
timeline.replaceChildren();let ts=data.timeline.filter(x=>x.run_id===c).sort((a,b)=>a.start_ns-b.start_ns).slice(0,400);timeline.setAttribute('viewBox',`0 0 1100 ${Math.max(60,ts.length*20+30)}`);if(!ts.length)return;
const start=Math.min(...ts.map(x=>x.start_ns)),end=Math.max(...ts.map(x=>x.end_ns)),range=Math.max(1,end-start);ts.forEach((x,i)=>{timeline.append(node('text',{x:5,y:i*20+15,'font-size':10},x.phase+' '+x.agent_id));let r=node('rect',{x:180+900*(x.start_ns-start)/range,y:i*20+3,width:Math.max(1,900*(x.end_ns-x.start_ns)/range),height:14,fill:x.outcome==='ok'?'#397bb8':'#c45b49'});r.append(node('title',{},`${x.round_id} ${x.agent_id}: ${x.duration_ms.toFixed(1)} ms (${x.outcome})`));timeline.append(r)})}
cell.addEventListener('change',draw);draw();</script></html>"""
