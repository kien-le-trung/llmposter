# Experiments

This folder stores benchmark datasets, component configs, composed run configs,
and experiment runners.

## Layout

- `artifacts/backend/`: retained backend diagnostic runs, latency pilots, test outputs, and tool caches.
- `artifacts/root/`: pytest temporary folders and tool caches moved from the repository root.
- `datasets/`: fixed benchmark datasets.
- `run_configs/`: full run configs and component configs.
- `run_configs/llm_configs/`: LLM provider/model components.
- `run_configs/embedding_configs/`: embedding model components.
- `run_configs/eval_dataset_configs/`: dataset selection components.
- `run_configs/prompt_configs/`: prompt technique components.
- `run_configs/voting_algo_configs/`: voting algorithm components.
- `src/runners/`: experiment runner and config composition scripts.

## Runners

Start the backend with a run config:

```powershell
python experiments/src/runners/run_backend.py --config qwen_7b
```

Compose a run config from components:

```powershell
python experiments/src/runners/compose_run_config.py --llm qwen_7b --embedding nomic_embed_text --prompt few_shot --eval-dataset standard_benchmark --voting-algo llm
```

Live rounds always include timed discussion and structured LLM voting, including
AI-only rounds (`include_human: false`). Clues use the backend's LangGraph pipeline.
The runner waits for discussion to close before explicitly requesting votes.
Its default polling timeout is 180 seconds; use a longer timeout when configuring
longer discussions or slower inference. Embedding components remain available for
offline workflows and do not select the live voting algorithm.

## AI-only latency benchmark

Run from the repository root using the backend virtual environment. The dedicated
runner starts a single-worker backend for each configuration, creates AI-only
rounds over HTTP, polls through discussion, and explicitly submits votes. It does
not need the frontend. Existing sequential experiments are unchanged.

Quick controlled comparison (ten games, approximately a minute or two):

```powershell
backend/venv/Scripts/python.exe experiments/src/runner/run_latency.py --capacities 1,4 --parallel-rounds 1,4 --delays 0.2 --skip-client-sweep --repetitions 1 --discussion-seconds 8 --no-mlflow
```

Full controlled matrix:

```powershell
backend/venv/Scripts/python.exe experiments/src/runner/run_latency.py
```

Controlled mode is the default. It launches an asynchronous OpenAI-compatible
stub with 200 ms and 1 second delays, uses isolated SQLite test databases, and
runs 15-second discussions. Five repetitions sweep discussion capacities
1/2/4/8 against 1/4 concurrent rounds, plus a separate client-limit sweep of 1/4
with discussion capacity held at four. This full matrix runs 300 games and takes
substantially longer than the quick comparison. `--skip-client-sweep`,
`--capacities`, `--parallel-rounds`, `--client-limits`, `--delays`,
`--repetitions`, and `--discussion-seconds` narrow or extend it.

Use `--scenario rate_limit`, `invalid`, `silence`, `timeout`, or `deadline` for
fault experiments. The first rate-limit, invalid-output, or timeout fault occurs
once per configuration; silence applies to discussion decisions, and deadline
holds discussion responses beyond the deadline. Timeout uses a 21-second delay
to exceed the real inference client's 20-second timeout. These are separate fault
tests, not comparable normal-latency samples. Very short discussions can expire
while HTTP clients are still being constructed; 15 seconds is the controlled
default for that reason.

The paid OpenRouter pilot is explicitly selected:

```powershell
backend/venv/Scripts/python.exe experiments/src/runner/run_latency.py --mode live --llm-config experiments/run_configs/llm_configs/openrouter.json
```

Live mode requires configured OpenRouter credentials and credits. It holds the
model constant, runs 90-second discussions, and randomizes capacities 1/4/8
against 1/4 concurrent rounds: 15 measured games plus one excluded warm-up game.
Use `--model` to override the configured paid model. Routing and token usage are
recorded when returned by the provider. The runner never migrates model settings.
More repetitions cost additional inference calls; this pilot is descriptive.

By default, both modes create independent local test databases, so persistence
timing represents SQLite rather than production PostgreSQL. To measure a
dedicated PostgreSQL test database, set its URL in an environment variable and
pass `--database-url-env LATENCY_TEST_DATABASE_URL`. The runner initializes
additive tables in that database; do not point it at production. Credentials are
not written to benchmark configuration artifacts.

Each run writes under `experiments/latency/results/<run>/` (override with
`--output`):

- `report.html`: standalone interactive tables, p95 queue/HTTP comparison, and
  request-overlap timeline. Open it directly; no web server is needed.
- `events.jsonl`: correlated background timing records across all configurations.
- `rounds.csv`, `turns.csv`, `messages.csv`: phase, agent-turn, publication, and
  first-observation timings. Clues include persistence and ordered-reveal times.
- `params.json`, `metrics.json`: sampling settings, matrix, summaries, failure
  counts, throughput, and outcome rates.
- Per-configuration backend/stub logs, settings, events, and isolated database.

MLflow tracking is enabled by default and uses a local database in the results
directory; `--no-mlflow` keeps the same local reports without tracking. Tracking
failures preserve reports and create `tracking_error.json`. Partial results are
saved after every cell, including failures. Use the process exit status and
`failed_phase`/`error_type` columns to distinguish failed games from completions.
If a batch fails, the runner skips the remaining repetitions of that cell and
stops its backend so unfinished rounds cannot contaminate the next measurements.

Timing events contain identifiers and numeric metadata, not prompts, private
memory, response text, or credentials. Normal backend startup does not enable
this telemetry, and private graph tracing remains disabled.

Interpret total game latency alongside the fixed discussion timer. Shared
discussion capacity is a backend-wide limit; model request limits are per
inference client. HTTP timing excludes the local semaphore wait but includes
network and upstream queueing, so it is not pure model compute time. Scheduler
wait includes deliberate debounce/cooldown/silence waits; `scheduler_dispatch_ms`
isolates the small interval after a turn becomes eligible. Silent, expired,
cancelled, and failed turns remain in the results. Percentiles use nearest rank
and are marked as small samples below 20 observations. Outcome-specific rows
separate successful completions from durations cut short by cancellation.


### Concurrency pilot

The runner now warms every configuration and records batch indexes and source/environment metadata.
Live execution requires a non-resetting OpenRouter key with no more than $5 remaining; the
runner checks this before inference. Use a dedicated key for the pilot.

Controlled pilot (100 measured games):

```powershell
backend/venv/Scripts/python.exe experiments/src/runner/run_latency.py --capacities 1,4 --parallel-rounds 1,4 --delays 0.2,1 --skip-client-sweep --repetitions 5 --discussion-seconds 15 --no-mlflow
```

Live pilot (10 measured games plus excluded warm-ups):

```powershell
backend/venv/Scripts/python.exe experiments/src/runner/run_latency.py --mode live --capacities 1,4 --parallel-rounds 1 --skip-client-sweep --repetitions 5 --discussion-seconds 30 --no-mlflow
```

`opening_measurements.json` retains missing speakers as null. `comparisons.json` and
`findings.md` report matched serial/concurrent opening comparisons, averaging rounds within
batches before taking medians. Incomplete batches are excluded from speedup calculations,
so inspect completion counts too. Throughput excludes gaps between discussion intervals.
Both conditions use asyncio; these experiments measure request overlap, not runtime choice.
Five batches provide descriptive evidence, especially for tail latency.
