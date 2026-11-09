# Concurrency latency pilot

Descriptive results; incomplete openings are excluded from latency comparisons and retained in opening_measurements.json.


Speedups use the median of batch-average complete opening latencies; a batch is excluded if any round has an incomplete opening. Missing comparisons indicate insufficient complete batches, not zero latency.


Both conditions use asyncio. This measures request overlap, not a runtime comparison. Fixed discussion timers mask total-game speedup. SQLite, provider routing, silence, retries, and five-batch sampling limit generalization.
