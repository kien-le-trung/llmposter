# Experiments

This folder stores benchmark datasets, component configs, composed run configs,
and experiment runners.

## Layout

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
python experiments/src/runners/compose_run_config.py --llm qwen_7b --embedding nomic_embed_text --prompt few_shot --eval-dataset standard_benchmark --voting-algo embedding_distance_v1
```
