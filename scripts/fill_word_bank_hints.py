from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_WORD_BANK = Path(__file__).resolve().parents[1] / "backend" / "app" / "data" / "word_bank.json"
DEFAULT_URL = "http://localhost:8888/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GENERIC_HINTS = {
    "action",
    "concept",
    "entity",
    "item",
    "object",
    "person",
    "place",
    "stuff",
    "thing",
    "word",
}


class ModelResponseError(Exception):
    """Raised when the model server responds but the content is unusable."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill word_bank.json hints with a local LLM.")
    parser.add_argument("--word-bank", type=Path, default=DEFAULT_WORD_BANK)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if args.skip_first < 0:
        raise SystemExit("--skip-first must be 0 or greater")

    entries = _load_entries(args.word_bank)
    skip_count = min(args.skip_first, len(entries))
    filled_entries: list[dict[str, str]] = entries[:skip_count]
    if skip_count:
        print(f"Skipped {skip_count}/{len(entries)}")

    for start in range(skip_count, len(entries), args.batch_size):
        batch = entries[start : start + args.batch_size]
        words = [entry["word"] for entry in batch]
        fallback_hints = {entry["word"]: entry["hint"] for entry in batch}
        hints = _generate_hints(args.url, args.model, words, fallback_hints)
        filled_entries.extend(
            {"word": word, "hint": hints[word]}
            for word in words
        )
        print(f"Filled {min(start + len(batch), len(entries))}/{len(entries)}")
        _write_entries(args.word_bank, filled_entries + entries[start + len(batch) :])
        if args.sleep > 0:
            time.sleep(args.sleep)

    _write_entries(args.word_bank, filled_entries)


def _load_entries(path: Path) -> list[dict[str, str]]:
    raw_entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_entries, list):
        raise ValueError("word bank must be a JSON list")

    entries: list[dict[str, str]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("word"), str):
            raise ValueError(f"entry {index} must contain a word string")

        word = entry["word"].strip().lower()
        if not word:
            raise ValueError(f"entry {index} has an empty word")

        existing_hint = entry.get("hint", "")
        hint = _normalize_hint(existing_hint, word) if isinstance(existing_hint, str) else None
        entries.append({"word": word, "hint": hint or _fallback_hint(word)})

    return entries


def _write_entries(path: Path, entries: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _generate_hints(
    url: str,
    model: str,
    words: list[str],
    fallback_hints: dict[str, str],
) -> dict[str, str]:
    if not words:
        return {}

    try:
        hints = _request_hints(url, model, words)
    except ModelResponseError as exc:
        return _recover_from_bad_response(url, model, words, fallback_hints, str(exc))

    invalid_words = [
        word for word in words if _normalize_hint(hints.get(word, ""), word) is None
    ]
    if not invalid_words:
        return {
            word: _normalize_hint(hints[word], word) or ""
            for word in words
        }

    if len(words) == 1:
        word = words[0]
        return _fallback_result(word, fallback_hints, f"model returned {hints.get(word)!r}")

    midpoint = len(words) // 2
    return {
        **_generate_hints(url, model, words[:midpoint], fallback_hints),
        **_generate_hints(url, model, words[midpoint:], fallback_hints),
    }


def _recover_from_bad_response(
    url: str,
    model: str,
    words: list[str],
    fallback_hints: dict[str, str],
    reason: str,
) -> dict[str, str]:
    if len(words) == 1:
        return _fallback_result(words[0], fallback_hints, reason)

    midpoint = len(words) // 2
    print(f"Retrying smaller batches for {len(words)} words; {reason}")
    return {
        **_generate_hints(url, model, words[:midpoint], fallback_hints),
        **_generate_hints(url, model, words[midpoint:], fallback_hints),
    }


def _fallback_result(word: str, fallback_hints: dict[str, str], reason: str) -> dict[str, str]:
    fallback = _normalize_hint(fallback_hints.get(word, ""), word) or _fallback_hint(word)
    print(f"Using fallback hint for {word!r}; {reason}")
    return {word: fallback}


def _request_hints(url: str, model: str, words: list[str]) -> dict[str, str]:
    payload = {
        "model": model,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": max(256, len(words) * 12),
        "messages": [
            {
                "role": "system",
                "content": "You create one-word hint labels for a word guessing game.",
            },
            {
                "role": "user",
                "content": _build_prompt(words),
            },
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach model server at {url}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
        parsed = _parse_first_json_object(content)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ModelResponseError(f"invalid response JSON: {exc}") from exc

    hints = parsed.get("hints")
    if not isinstance(hints, dict):
        raise ModelResponseError("model response must contain a hints object")

    normalized_hints = {
        str(word).strip().lower(): str(hint).strip().lower()
        for word, hint in hints.items()
    }
    if len(words) == 1 and words[0] not in normalized_hints and len(normalized_hints) == 1:
        return {words[0]: next(iter(normalized_hints.values()))}

    return normalized_hints


def _build_prompt(words: list[str]) -> str:
    word_lines = "\n".join(f"- {word}" for word in words)
    return (
        "For each game word, generate exactly one associated hint word.\n"
        "The hint is shown to an imposter who does not know the secret word.\n"
        "Rules:\n"
        "- hint must be exactly one lowercase English word\n"
        "- hint must not be the same as the game word\n"
        "- hint must not contain the game word\n"
        "- prefer a broad category or strong association\n"
        "- avoid generic hints like thing, object, item, place, person, concept\n"
        "- return JSON only with this exact shape: {\"hints\":{\"word\":\"hint\"}}\n\n"
        "Words:\n"
        f"{word_lines}"
    )


def _parse_first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    start_index = stripped.find("{")
    if start_index == -1:
        raise ValueError("no JSON object found in model response")

    parsed, _ = json.JSONDecoder().raw_decode(stripped[start_index:])
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON must be an object")

    return parsed


def _normalize_hint(value: str, word: str) -> str | None:
    match = re.fullmatch(r"[a-z][a-z'-]*", value.strip().lower())
    if match is None:
        return None

    hint = match.group(0)
    if hint == word or hint in word or word in hint:
        return None
    if hint in GENERIC_HINTS:
        return None

    return hint


def _fallback_hint(word: str) -> str:
    if word != "theme":
        return "theme"
    return "clue"


if __name__ == "__main__":
    main()
