# Prompt Benchmarking Implementation Notes

Branch: `prompt-benchmarking`

## Summary

Implemented the first prompt-technique benchmarking layer for clue generation. The implementation keeps production prompts in `/backend` and reserves `/experiments` for benchmark scripts, fixed benchmark data, result files, and future MLflow integration.

## Backend changes

### `backend/app/core/config.py`

Added:

```python
clue_prompt_technique: str = Field(default="few_shot", alias="CLUE_PROMPT_TECHNIQUE")
```

The default prompt technique is `few_shot`.

### `backend/app/prompts/strategies.json`

Changed strategy definitions from one prompt per strategy to four complete prompt variants per strategy:

- `zero_shot`
- `few_shot`
- `reasoning_guided`
- `meta`

This applies to every non-imposter and imposter strategy.

### `backend/app/prompts/strategy_loader.py`

Updated the strategy loader to:

- validate all four prompt variants for every strategy,
- normalize unknown techniques to `few_shot`,
- expose `load_non_imposter_clue_strategies(...)`,
- expose `load_imposter_clue_strategies(...)`,
- keep `NON_IMPOSTER_CLUE_STRATEGIES` and `IMPOSTER_CLUE_STRATEGIES` compatible with the existing `random.choice(...)` route code through a sequence facade.

### `backend/app/prompts/__init__.py`

Exported prompt-technique constants and strategy loader functions.

### `backend/app/services/agents.py`

Re-exported prompt-technique constants and strategy loader functions from the agent service boundary.

### `backend/tests/test_strategy_prompt_techniques.py`

Added tests covering:

- `few_shot` default behavior,
- all technique variants loading for non-imposter strategies,
- all technique variants loading for imposter strategies,
- selected prompt text changing across techniques,
- strategy constants preserving the existing `{"name", "prompt"}` shape.

## Experiment changes

### `experiments/prompt_benchmarking/benchmark_cases.json`

Added 30 fixed benchmark cases copied from the committed backend word bank, each with a fixed human clue.

### `experiments/prompt_benchmarking/schemas.py`

Added Pydantic schemas for benchmark cases, benchmark config, round records, clue records, and summary metrics.

### `experiments/prompt_benchmarking/api_client.py`

Added an async backend API client that calls existing backend endpoints:

- `POST /rounds`
- `GET /rounds/{round_id}`
- `POST /rounds/{round_id}/clue` when the round waits for human input

The client measures round-level latency and extracts generated agent clues.

### `experiments/prompt_benchmarking/metrics.py`

Added summary metrics:

- average latency only,
- round success rate,
- generation failed rate,
- duplicate clue rate,
- secret word leak rate,
- empty clue rate,
- average clue word count.

No `num_rounds`, `num_clues`, p50 latency, or p95 latency metrics are computed.

### `experiments/prompt_benchmarking/benchmark_prompts.py`

Added the CLI runner. It treats the backend as a service and writes JSONL/JSON result files.

### `experiments/prompt_benchmarking/test_metrics.py`

Added tests for average latency, failed status rate, word reuse metric, duplicate clues, and empty inputs.

### `experiments/prompt_benchmarking/README.md`

Documented setup and usage.

### `.gitignore`

Ignored generated benchmark output files and local `mlruns/`, while keeping `results/.gitkeep` tracked.

## Intentional non-changes

No changes were made to:

- frontend code,
- database schema,
- voting algorithm,
- Docker Compose,
- deployment config,
- production API response models,
- MLflow integration.

MLflow is intentionally deferred so the benchmark runner works independently first.
