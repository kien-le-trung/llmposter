from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_DIR / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_DIR / "experiments" / "datasets"

CLUE_OUTPUT_FIELDS = [
    "clue_id",
    "round_id",
    "player_id",
    "player_name",
    "clue",
    "source",
    "inference_mode",
    "sequence",
    "created_at",
    "word_count",
    "character_count",
    "normalized_clue",
    "round_has_human",
    "round_clue_count",
]

VOTE_OUTPUT_FIELDS = [
    "round_id",
    "human_voted_for_id",
    "human_voted_for_name",
    "human_selected_clue",
    "human_selected_source",
    "machine_majority_voted_for_id",
    "machine_majority_voted_for_name",
    "machine_majority_selected_clue",
    "machine_majority_selected_source",
    "machine_majority_status",
    "human_machine_majority_agree",
    "machine_vote_count",
    "machine_unique_target_count",
    "machine_leader_vote_count",
    "machine_leader_vote_share",
    "machine_vote_counts_json",
    "recorded_group_voted_for_id",
    "recorded_group_voted_for_name",
    "recorded_group_selected_clue",
    "recorded_group_selected_source",
    "recorded_group_status",
    "human_recorded_group_agree",
    "combined_vote_counts_json",
    "imposter_won",
    "round_winner",
    "inferred_imposter_id",
    "human_vote_correct",
    "created_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build analysis datasets from raw Neon clue and vote exports."
    )
    parser.add_argument("--clues", type=Path)
    parser.add_argument("--votes", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clues_path = args.clues or _latest_export(args.raw_dir, "clues_*.csv")
    votes_path = args.votes or _latest_export(args.raw_dir, "votes_*.csv")
    clue_rows = _read_csv(clues_path)
    vote_rows = _read_csv(votes_path)
    _validate_rows(clue_rows, vote_rows)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clue_output = output_dir / "human_machine_clues.csv"
    vote_output = output_dir / "human_voting_patterns.csv"
    rounds_output = output_dir / "human_behavior_rounds.jsonl"
    manifest_output = output_dir / "human_behavior_manifest.json"

    processed_clues = _process_clues(clue_rows)
    processed_votes = _process_votes(vote_rows, clue_rows)
    _write_csv(clue_output, CLUE_OUTPUT_FIELDS, processed_clues)
    _write_csv(vote_output, VOTE_OUTPUT_FIELDS, processed_votes)
    _write_rounds_jsonl(rounds_output, clue_rows, vote_rows)

    raw_clue_rounds = {row["round_id"] for row in clue_rows}
    human_clue_rounds = {
        row["round_id"] for row in clue_rows if row["player_id"] == "human"
    }
    voting_rounds = {row["round_id"] for row in processed_votes}
    machine_agreement_count = sum(
        row["human_machine_majority_agree"] == "1" for row in processed_votes
    )
    machine_disagreement_count = sum(
        row["human_machine_majority_agree"] == "0" for row in processed_votes
    )
    manifest = {
        "dataset_version": _shared_export_version(clues_path, votes_path),
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "clues": _source_metadata(clues_path),
            "votes": _source_metadata(votes_path),
        },
        "outputs": {
            "clues": clue_output.name,
            "voting": vote_output.name,
            "rounds": rounds_output.name,
        },
        "counts": {
            "clues": len(processed_clues),
            "human_clues": sum(row["source"] == "human" for row in processed_clues),
            "machine_clues": sum(row["source"] == "machine" for row in processed_clues),
            "clue_rounds": len(raw_clue_rounds),
            "human_participation_rounds": len(human_clue_rounds),
            "complete_voting_rounds": len(voting_rounds),
            "human_machine_majority_agreements": machine_agreement_count,
            "human_machine_majority_disagreements": machine_disagreement_count,
            "machine_majority_ties": sum(
                row["machine_majority_status"] == "tie" for row in processed_votes
            ),
            "recorded_group_ties": sum(
                row["recorded_group_status"] == "tie" for row in processed_votes
            ),
        },
        "limitations": [
            "The database did not persist secret_word, imposter_hint, playing order, or explicit imposter identity.",
            "Clue created_at order is not guaranteed to equal the order visible to players because machine clues may be generated concurrently.",
            "The imposter identity is inferable only when the aggregate vote caught the imposter (imposter_won=false).",
            "Human vote correctness is therefore unknown for rounds won by the imposter.",
            "The recorded group tally includes the human vote; machine-majority fields are recomputed from machine votes only.",
        ],
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote {clue_output} ({len(processed_clues)} rows)")
    print(f"Wrote {vote_output} ({len(processed_votes)} rows)")
    print(f"Wrote {rounds_output} ({len(raw_clue_rounds | voting_rounds)} rounds)")
    print(f"Wrote {manifest_output}")


def _latest_export(raw_dir: Path, pattern: str) -> Path:
    candidates = sorted(raw_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} in {raw_dir}")
    return candidates[-1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_rows(
    clues: list[dict[str, str]], votes: list[dict[str, str]]
) -> None:
    if not clues or not votes:
        raise ValueError("Clue and vote exports must both contain data.")
    required_clues = {"id", "round_id", "player_id", "clue", "created_at"}
    required_votes = {
        "id",
        "round_id",
        "voter_id",
        "voted_for_id",
        "imposter_won",
        "round_winner",
    }
    missing_clues = required_clues - clues[0].keys()
    missing_votes = required_votes - votes[0].keys()
    if missing_clues or missing_votes:
        raise ValueError(
            f"Missing required fields: clues={sorted(missing_clues)}, "
            f"votes={sorted(missing_votes)}"
        )
    _assert_unique(clues, ("round_id", "player_id", "sequence"), "clues")
    _assert_unique(votes, ("round_id", "voter_id"), "votes")


def _assert_unique(
    rows: list[dict[str, str]], fields: tuple[str, ...], label: str
) -> None:
    keys = [tuple(row.get(field, "") for field in fields) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate {label} keys for {fields}: {duplicates[:3]}")


def _process_clues(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_round: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_round.setdefault(row["round_id"], []).append(row)

    processed: list[dict[str, Any]] = []
    for row in rows:
        round_rows = by_round[row["round_id"]]
        clue = row["clue"].strip()
        processed.append(
            {
                "clue_id": row["id"],
                "round_id": row["round_id"],
                "player_id": row["player_id"],
                "player_name": row.get("player_name", ""),
                "clue": clue,
                "source": "human" if row["player_id"] == "human" else "machine",
                "inference_mode": row.get("inference_mode", ""),
                "sequence": row.get("sequence", ""),
                "created_at": row["created_at"],
                "word_count": len(re.findall(r"\b[\w'-]+\b", clue)),
                "character_count": len(clue),
                "normalized_clue": " ".join(clue.casefold().split()),
                "round_has_human": _bool_string(
                    any(item["player_id"] == "human" for item in round_rows)
                ),
                "round_clue_count": len(round_rows),
            }
        )
    return processed


def _process_votes(
    votes: list[dict[str, str]], clues: list[dict[str, str]]
) -> list[dict[str, Any]]:
    votes_by_round: dict[str, list[dict[str, str]]] = {}
    clues_by_target = {
        (row["round_id"], row["player_id"]): row for row in clues
    }
    for row in votes:
        votes_by_round.setdefault(row["round_id"], []).append(row)

    processed: list[dict[str, Any]] = []
    for round_id, round_votes in sorted(votes_by_round.items()):
        by_voter = {row["voter_id"]: row for row in round_votes}
        human = by_voter.get("human")
        group = by_voter.get("group")
        if human is None or group is None:
            continue

        agent_votes = [
            row for row in round_votes if row["voter_id"] not in {"human", "group"}
        ]
        machine_vote_counts = Counter(row["voted_for_id"] for row in agent_votes)
        machine_highest = max(machine_vote_counts.values(), default=0)
        machine_leaders = sorted(
            target
            for target, count in machine_vote_counts.items()
            if count == machine_highest
        )
        machine_majority_id = machine_leaders[0] if len(machine_leaders) == 1 else ""
        machine_majority_status = (
            "selected" if machine_majority_id else "tie" if machine_leaders else "no_votes"
        )
        combined_vote_counts = machine_vote_counts.copy()
        if human["voted_for_id"]:
            combined_vote_counts[human["voted_for_id"]] += 1
        human_target = clues_by_target.get((round_id, human["voted_for_id"]))
        machine_target = clues_by_target.get((round_id, machine_majority_id))
        group_target = clues_by_target.get((round_id, group["voted_for_id"]))
        machine_agree = (
            human["voted_for_id"] == machine_majority_id
            if machine_majority_id
            else None
        )
        group_agree = (
            human["voted_for_id"] == group["voted_for_id"]
            if group["voted_for_id"]
            else None
        )
        group_status = "selected" if group["voted_for_id"] else "tie"
        imposter_won = _parse_optional_bool(group.get("imposter_won", ""))
        inferred_imposter_id = group["voted_for_id"] if imposter_won is False else ""
        human_correct = group_agree if imposter_won is False else None

        processed.append(
            {
                "round_id": round_id,
                "human_voted_for_id": human["voted_for_id"],
                "human_voted_for_name": human.get("voted_for_name", ""),
                "human_selected_clue": human_target["clue"] if human_target else "",
                "human_selected_source": _clue_source(human_target),
                "machine_majority_voted_for_id": machine_majority_id,
                "machine_majority_voted_for_name": (
                    machine_target.get("player_name", "") if machine_target else ""
                ),
                "machine_majority_selected_clue": (
                    machine_target["clue"] if machine_target else ""
                ),
                "machine_majority_selected_source": _clue_source(machine_target),
                "machine_majority_status": machine_majority_status,
                "human_machine_majority_agree": _optional_bool_string(machine_agree),
                "machine_vote_count": len(agent_votes),
                "machine_unique_target_count": len(machine_vote_counts),
                "machine_leader_vote_count": machine_highest,
                "machine_leader_vote_share": (
                    f"{machine_highest / len(agent_votes):.6f}" if agent_votes else ""
                ),
                "machine_vote_counts_json": json.dumps(
                    dict(sorted(machine_vote_counts.items())), ensure_ascii=False
                ),
                "recorded_group_voted_for_id": group["voted_for_id"],
                "recorded_group_voted_for_name": group.get("voted_for_name", ""),
                "recorded_group_selected_clue": group_target["clue"]
                if group_target
                else "",
                "recorded_group_selected_source": _clue_source(group_target),
                "recorded_group_status": group_status,
                "human_recorded_group_agree": _optional_bool_string(group_agree),
                "combined_vote_counts_json": json.dumps(
                    dict(sorted(combined_vote_counts.items())), ensure_ascii=False
                ),
                "imposter_won": _optional_bool_string(imposter_won),
                "round_winner": group.get("round_winner", ""),
                "inferred_imposter_id": inferred_imposter_id,
                "human_vote_correct": _optional_bool_string(human_correct),
                "created_at": group.get("created_at", ""),
            }
        )
    return processed


def _write_rounds_jsonl(
    path: Path, clues: list[dict[str, str]], votes: list[dict[str, str]]
) -> None:
    clues_by_round: dict[str, list[dict[str, str]]] = {}
    votes_by_round: dict[str, list[dict[str, str]]] = {}
    for row in clues:
        clues_by_round.setdefault(row["round_id"], []).append(row)
    for row in votes:
        votes_by_round.setdefault(row["round_id"], []).append(row)

    round_ids = sorted(clues_by_round.keys() | votes_by_round.keys())
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for round_id in round_ids:
            payload = {
                "round_id": round_id,
                "clues": clues_by_round.get(round_id, []),
                "votes": votes_by_round.get(round_id, []),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _source_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_DIR)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _shared_export_version(clues_path: Path, votes_path: Path) -> str:
    clue_match = re.search(r"clues_(.+)\.csv$", clues_path.name)
    vote_match = re.search(r"votes_(.+)\.csv$", votes_path.name)
    clue_version = clue_match.group(1) if clue_match else clues_path.stem
    vote_version = vote_match.group(1) if vote_match else votes_path.stem
    return clue_version if clue_version == vote_version else f"{clue_version}__{vote_version}"


def _clue_source(row: dict[str, str] | None) -> str:
    if row is None:
        return ""
    return "human" if row["player_id"] == "human" else "machine"


def _parse_optional_bool(value: str) -> bool | None:
    normalized = value.strip().casefold()
    if normalized in {"true", "t", "1", "yes"}:
        return True
    if normalized in {"false", "f", "0", "no"}:
        return False
    return None


def _bool_string(value: bool) -> str:
    return "1" if value else "0"


def _optional_bool_string(value: bool | None) -> str:
    return "" if value is None else _bool_string(value)


if __name__ == "__main__":
    main()
