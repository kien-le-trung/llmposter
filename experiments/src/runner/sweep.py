from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

from .config import SingleExperimentConfig, SweepConfig, parse_single_config


def expand_sweep(sweep_config: SweepConfig) -> list[SingleExperimentConfig]:
    keys = list(sweep_config.vary)
    value_grid = [sweep_config.vary[key] for key in keys]
    variants: list[SingleExperimentConfig] = []

    for index, values in enumerate(product(*value_grid)):
        variant_data = deepcopy(sweep_config.base)
        sweep_values = dict(zip(keys, values, strict=True))
        for dotted_key, value in sweep_values.items():
            _deep_set(variant_data, dotted_key, value)

        variant_name = f"{sweep_config.name}_{index + 1:03d}"
        variant_data["name"] = variant_name
        variant_data["sweep_name"] = sweep_config.name
        variant_data["sweep_variant_index"] = index
        variant_data["sweep_values"] = sweep_values

        variants.append(
            parse_single_config(
                variant_data,
                source_path=sweep_config.source_path,
            )
        )

    return variants


def _deep_set(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid sweep key: {dotted_key!r}")

    current = target
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            raise ValueError(f"Sweep key {dotted_key!r} cannot descend into {part!r}")
        current = next_value
    current[parts[-1]] = value
