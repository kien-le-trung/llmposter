from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "datasets" / "scraped"
RAW_RESULTS_FILENAME = "raw_results.jsonl"

CSV_FIELDS = [
    "experiment_name",
    "case_id",
    "repetition_index",
    "round_id",
    "turn_id",
    "turn_sequence",
    "secret_word",
    "imposter_was",
    "candidate_agent_id",
    "candidate_clue",
    "is_imposter",
    "all_clues_json",
]


@dataclass(frozen=True)
class ScrapeResult:
    rows: list[dict[str, str]]
    source_path: Path


class RawResultScraper:
    def __init__(
        self,
        *,
        output_root: Path | None = None,
        dataset_dir: Path | None = None,
    ) -> None:
        self.output_root = output_root or REPO_DIR / "experiments" / "output"
        self.dataset_dir = dataset_dir or DEFAULT_OUTPUT_DIR

    def scrape_experiment(self, experiment_name: str) -> ScrapeResult:
        source_path = self.output_root / experiment_name / RAW_RESULTS_FILENAME
        if not source_path.exists():
            raise FileNotFoundError(f"Raw results file not found: {source_path}")

        rows: list[dict[str, str]] = []
        for line_number, payload in _read_jsonl(source_path):
            rows.extend(
                self._scrape_result(
                    experiment_name=experiment_name,
                    payload=payload,
                    line_number=line_number,
                )
            )

        return ScrapeResult(rows=rows, source_path=source_path)

    def export(
        self,
        *,
        experiment_name: str,
        dataset_version: str | None = None,
    ) -> tuple[Path, Path]:
        result = self.scrape_experiment(experiment_name)
        version = dataset_version or _default_dataset_version(experiment_name)

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.dataset_dir / f"voting_candidates_{version}.csv"
        manifest_path = self.dataset_dir / f"voting_candidates_{version}.manifest.json"

        _write_csv(csv_path, result.rows)
        _write_manifest(
            manifest_path,
            {
                "dataset_version": version,
                "experiment_name": experiment_name,
                "source_path": str(result.source_path),
                "row_count": len(result.rows),
                "fieldnames": CSV_FIELDS,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

        return csv_path, manifest_path

    def _scrape_result(
        self,
        *,
        experiment_name: str,
        payload: dict[str, Any],
        line_number: int,
    ) -> list[dict[str, str]]:
        request = _as_mapping(payload.get("request"))
        round_payload = _as_mapping(payload.get("round_payload"))
        vote_payload = _as_mapping(payload.get("vote_payload"))

        if not round_payload:
            return []

        secret_word = _string_value(request.get("secret_word"))
        imposter_was = _string_value(vote_payload.get("imposter_was"))
        round_id = _string_value(payload.get("round_id") or round_payload.get("id"))
        case_id = _string_value(request.get("case_id"))
        repetition_index = _string_value(request.get("repetition_index"))

        rows: list[dict[str, str]] = []
        for turn in _iter_turns(round_payload):
            responses = _extract_agent_responses(turn)
            if not responses:
                continue

            all_clues = [
                {
                    "agent_id": response["agent_id"],
                    "agent_name": response["agent_name"],
                    "clue": response["clue"],
                }
                for response in responses
            ]
            all_clues_json = json.dumps(all_clues, ensure_ascii=False)

            for response in responses:
                candidate_agent_name = response["agent_name"]
                rows.append(
                    {
                        "experiment_name": experiment_name,
                        "case_id": case_id,
                        "repetition_index": repetition_index,
                        "round_id": round_id,
                        "turn_id": _string_value(turn.get("id")),
                        "turn_sequence": _string_value(turn.get("sequence")),
                        "secret_word": secret_word,
                        "imposter_was": imposter_was,
                        "candidate_agent_id": response["agent_id"],
                        "candidate_clue": response["clue"],
                        "is_imposter": _bool_label(candidate_agent_name == imposter_was)
                        if imposter_was
                        else "",
                        "all_clues_json": all_clues_json,
                    }
                )

        if not rows:
            raise ValueError(
                f"No agent clues found in {experiment_name} line {line_number}."
            )

        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract voting candidate rows from experiment raw_results.jsonl."
    )
    parser.add_argument("experiment_name")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_DIR / "experiments" / "output",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scraper = RawResultScraper(
        output_root=args.output_root,
        dataset_dir=args.dataset_dir,
    )
    csv_path, manifest_path = scraper.export(
        experiment_name=args.experiment_name,
        dataset_version=args.dataset_version,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {manifest_path}")


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {path} line {line_number}.")
            rows.append((line_number, payload))
    return rows


def _iter_turns(round_payload: dict[str, Any]) -> list[dict[str, Any]]:
    turns = round_payload.get("turns")
    if not isinstance(turns, list):
        return []
    return [turn for turn in turns if isinstance(turn, dict)]


def _extract_agent_responses(turn: dict[str, Any]) -> list[dict[str, str]]:
    responses = turn.get("responses")
    if not isinstance(responses, list):
        return []

    extracted: list[dict[str, str]] = []
    for response in responses:
        if not isinstance(response, dict):
            continue
        agent_id = _string_value(response.get("agent_id"))
        if agent_id == "human":
            continue
        extracted.append(
            {
                "agent_id": agent_id,
                "agent_name": _string_value(response.get("agent_name")),
                "clue": _string_value(response.get("agent_response")),
            }
        )
    return extracted


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_value(value: Any) -> str:
    return "" if value is None else str(value)


def _bool_label(value: bool) -> str:
    return "1" if value else "0"


def _default_dataset_version(experiment_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in experiment_name.strip()
    ).strip("_")
    return f"{safe_name or 'dataset'}_{timestamp}"


if __name__ == "__main__":
    main()
