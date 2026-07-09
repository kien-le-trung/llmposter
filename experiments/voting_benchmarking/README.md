# Voting Benchmarking

This experiment package collects replayable round artifacts for future voting
algorithm benchmarks. The first version is also a difficulty snapshot: it records
how detectable the imposter is under the current backend clue generation and
embedding-based voting behavior.

## What This Collects

- Round metadata: word, hint, prompt technique, latency, status, playing order.
- Label metadata: inferred imposter player id, name, and player kind.
- Clues: clue text, player role, player position, inference mode, order features.
- Votes: current backend agent-only voting result recorded as `embedding_distance_v1`.
- Semantic features: optional embedding-based clue/word/hint similarity metrics.

The benchmark creates agent-only rounds, so every clue is LLM-generated. The API
currently exposes the true imposter as a name in the vote response, not as an
id. This benchmark infers `imposter_player_id` by matching that name against the
round's playing order, so player names should be unique.

## Running

Start the backend first, then run:

```powershell
python experiments/voting_benchmarking/benchmark_voting.py --technique few_shot --repetitions 2
```

The runner logs to MLflow, so install the backend dev dependencies first if
`mlflow` is not already available in the active Python environment.

Useful options:

```powershell
python experiments/voting_benchmarking/benchmark_voting.py --technique all --repetitions 20
python experiments/voting_benchmarking/benchmark_voting.py --show-progress --progress-every 5
python experiments/voting_benchmarking/benchmark_voting.py --skip-semantic-features
python experiments/voting_benchmarking/benchmark_voting.py --require-semantic-features
```

Semantic features use the backend `InferenceClient` embedding path. If embedding
extraction fails, the benchmark still writes round, clue, vote, and summary
artifacts unless `--require-semantic-features` is set.

## Outputs

Generated files are written under `experiments/voting_benchmarking/results/`:

- `round_artifacts_<technique>_<timestamp>.jsonl`
- `clues_<technique>_<timestamp>.jsonl`
- `votes_<technique>_<timestamp>.jsonl`
- `semantic_features_<technique>_<timestamp>.jsonl`
- `summary_<technique>_<timestamp>.json`

MLflow uses:

```text
sqlite:///experiments/voting_benchmarking/mlflow.db
```

and logs parameters, scalar summary metrics, position-sliced detection metrics,
and all generated artifacts.

## Interpretation

Primary detection metrics:

- `agent_vote_detection_rate`: share of individual agent votes that hit the imposter.
- `agent_only_group_detection_rate`: majority result among agent votes only.
- `group_detection_rate`: backend group result from agent votes only.
- `random_chance_detection_rate`: average `1 / num_players` baseline.
- `detection_lift_over_random`: agent-only group detection minus chance.

For agent-only benchmark runs, `agent_only_group_detection_rate` and
`group_detection_rate` are both computed from LLM-generated clues and agent
votes only. Tied group votes are excluded from `agent_only_group_detection_rate`
and recorded as null group decisions in the backend vote artifact.

Semantic features help separate game difficulty from voting quality:

- High non-imposter cohesion with low detection suggests voting weakness or a too-hard hint.
- Low non-imposter cohesion with low detection suggests clue generation noise.
- High imposter outlier score with low detection suggests the voting algorithm missed an obvious clue.
- Very high detection with high outlier score suggests the imposter is too obvious.
