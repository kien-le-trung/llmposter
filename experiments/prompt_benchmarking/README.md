# Prompt Benchmarking

This experiment package benchmarks clue-generation prompt techniques for LLMposter.

## What is being compared

The benchmark compares four `CLUE_PROMPT_TECHNIQUE` values:

- `zero_shot`
- `few_shot`
- `reasoning_guided`
- `meta`

The important design choice is that prompt technique variants live inside each strategy in `backend/app/prompts/strategies.json`. Each imposter and non-imposter strategy has a complete prompt for each technique.

## Backend setup

Start the backend with one technique at a time. Example:

```powershell
$env:CLUE_PROMPT_TECHNIQUE="few_shot"
cd backend
.\venv\Scripts\activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run the benchmark for the same technique from another terminal:

```powershell
python experiments/prompt_benchmarking/benchmark_prompts.py --techniques few_shot
```

Repeat by restarting the backend with another `CLUE_PROMPT_TECHNIQUE` value and rerunning the script for that technique.

The runner intentionally does not mutate backend configuration at runtime. This avoids adding benchmark-only production API routes.

## Benchmark cases

`benchmark_cases.json` contains 30 fixed word/hint pairs copied from `backend/app/data/word_bank.json`, plus a fixed human clue for each case. The set is fixed so results are comparable across prompt techniques.

## Outputs

Generated files are written under `experiments/prompt_benchmarking/results/` and ignored by git:

- `rounds_<timestamp>.jsonl`
- `clues_<timestamp>.jsonl`
- `summary_<timestamp>.json`

The summary currently records:

- average latency only
- round success rate
- generation failed rate
- duplicate clue rate
- secret word leak rate
- empty clue rate
- average clue word count

## Future MLflow integration

The benchmark is structured so MLflow can be added later as a thin wrapper around existing outputs:

1. Run benchmark cases.
2. Compute summary metrics.
3. Log params, metrics, and JSONL artifacts to MLflow.

The benchmark runner should remain usable without MLflow.
