# Concurrency latency pilot

Descriptive results; incomplete openings are excluded from latency comparisons and retained in opening_measurements.json.

- Opening completion for 00-discussion-cap4-rounds4-client4-delay0.2: 20/20 rounds.
- Opening completion for 01-discussion-cap1-rounds1-client4-delay1.0: 5/5 rounds.
- Opening completion for 02-discussion-cap4-rounds1-client4-delay1.0: 5/5 rounds.
- Opening completion for 03-discussion-cap4-rounds4-client4-delay1.0: 20/20 rounds.
- Opening completion for 04-discussion-cap4-rounds1-client4-delay0.2: 5/5 rounds.
- Opening completion for 05-discussion-cap1-rounds4-client4-delay1.0: 15/20 rounds.
- Opening completion for 06-discussion-cap1-rounds1-client4-delay0.2: 5/5 rounds.
- Opening completion for 07-discussion-cap1-rounds4-client4-delay0.2: 20/20 rounds.

Speedups use the median of batch-average complete opening latencies; a batch is excluded if any round has an incomplete opening. Missing comparisons indicate insufficient complete batches, not zero latency.

- 01-discussion-cap1-rounds1-client4-delay1.0 versus 02-discussion-cap4-rounds1-client4-delay1.0: 4124.1 ms versus 1111.6 ms; 3.71x speedup, 73.0% reduction.
- 06-discussion-cap1-rounds1-client4-delay0.2 versus 04-discussion-cap4-rounds1-client4-delay0.2: 968.4 ms versus 311.1 ms; 3.11x speedup, 67.9% reduction.
- 07-discussion-cap1-rounds4-client4-delay0.2 versus 00-discussion-cap4-rounds4-client4-delay0.2: 2253.1 ms versus 563.3 ms; 4.00x speedup, 75.0% reduction.

Both conditions use asyncio. This measures request overlap, not a runtime comparison. Fixed discussion timers mask total-game speedup. SQLite, provider routing, silence, retries, and five-batch sampling limit generalization.
