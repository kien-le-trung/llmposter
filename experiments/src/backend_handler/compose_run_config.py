from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_DIR / "experiments"
RUN_CONFIGS_DIR = EXPERIMENTS_DIR / "run_configs"

COMPONENT_DIRS = {
    "llm": RUN_CONFIGS_DIR / "llm_configs",
    "embedding": RUN_CONFIGS_DIR / "embedding_configs",
    "prompt": RUN_CONFIGS_DIR / "prompt_configs",
    "eval_dataset": RUN_CONFIGS_DIR / "eval_dataset_configs",
    "voting_algo": RUN_CONFIGS_DIR / "voting_algo_configs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose an experiment run config from component configs."
    )
    parser.add_argument("--llm", required=True, help="LLM component name or JSON path.")
    parser.add_argument(
        "--embedding",
        default="nomic_embed_text",
        help="Embedding component name or JSON path.",
    )
    parser.add_argument(
        "--prompt",
        default="few_shot",
        help="Prompt component name or JSON path.",
    )
    parser.add_argument(
        "--eval-dataset",
        default="standard_benchmark",
        help="Evaluation dataset component name or JSON path.",
    )
    parser.add_argument(
        "--voting-algo",
        default="embedding_distance_v1",
        help="Voting algorithm component name or JSON path.",
    )
    parser.add_argument(
        "--inference-mode",
        default="remote",
        help="Backend inference mode to include in the composed config.",
    )
    parser.add_argument(
        "--agent-config-source",
        default="static",
        help="Backend agent config source to include in the composed config.",
    )
    parser.add_argument(
        "--word-selection-mode",
        default="random",
        help="Backend word selection mode to include in the composed config.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = compose_run_config(
        llm=args.llm,
        embedding=args.embedding,
        prompt=args.prompt,
        eval_dataset=args.eval_dataset,
        voting_algo=args.voting_algo,
        inference_mode=args.inference_mode,
        agent_config_source=args.agent_config_source,
        word_selection_mode=args.word_selection_mode,
    )
    rendered = json.dumps(config, indent=2)

    if args.output is None:
        print(rendered)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")


def compose_run_config(
    *,
    llm: str,
    embedding: str,
    prompt: str,
    eval_dataset: str,
    voting_algo: str,
    inference_mode: str = "remote",
    agent_config_source: str = "static",
    word_selection_mode: str = "random",
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    component_specs = [
        ("llm", llm),
        ("embedding", embedding),
        ("prompt", prompt),
        ("eval_dataset", eval_dataset),
        ("voting_algo", voting_algo),
    ]

    for component_type, name_or_path in component_specs:
        component = load_component(component_type, name_or_path)
        _deep_merge(config, component)

    config["inference_mode"] = inference_mode
    config["agent_config_source"] = agent_config_source
    config["word_selection_mode"] = word_selection_mode
    return config


def load_component(component_type: str, name_or_path: str) -> dict[str, Any]:
    component_path = resolve_component_path(component_type, name_or_path)
    data = json.loads(component_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Component config {component_path} must contain a JSON object")
    return data


def resolve_component_path(component_type: str, name_or_path: str) -> Path:
    component_dir = COMPONENT_DIRS[component_type]
    path = Path(name_or_path)
    if path.suffix != ".json":
        path = path.with_suffix(".json")

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                Path.cwd() / path,
                REPO_DIR / path,
                component_dir / path.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"{component_type} component {name_or_path!r} not found. Searched: {searched}"
    )


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(target[key], value)
            continue

        target[key] = value


if __name__ == "__main__":
    main()
