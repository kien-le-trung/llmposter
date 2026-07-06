import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WORD_BANK_PATH = DATA_DIR / "word_bank.json"


@dataclass(frozen=True)
class WordBankEntry:
    word: str
    hint: str


class WordBankError(Exception):
    """Raised when the committed word bank cannot be used."""


def select_random_word(
    choice_func: Callable[[Sequence[WordBankEntry]], WordBankEntry] = random.choice,
) -> tuple[str, str]:
    entries = list_word_bank_entries()
    if not entries:
        raise WordBankError("Word bank is empty")

    entry = choice_func(entries)
    return entry.word, entry.hint


def list_word_bank_entries() -> list[WordBankEntry]:
    return list(_load_word_bank_entries())


def normalize_imposter_hint(hint: str) -> str:
    match = re.search(r"[A-Za-z0-9][A-Za-z0-9'-]*", hint)
    if match is None:
        return "thing"

    return match.group(0)


@lru_cache
def _load_word_bank_entries() -> tuple[WordBankEntry, ...]:
    try:
        raw_entries = json.loads(WORD_BANK_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WordBankError(f"Could not read word bank: {WORD_BANK_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise WordBankError(f"Word bank contains invalid JSON: {WORD_BANK_PATH}") from exc

    if not isinstance(raw_entries, list):
        raise WordBankError("Word bank must be a JSON list")

    entries = tuple(_parse_word_bank_entry(index, entry) for index, entry in enumerate(raw_entries))
    if not entries:
        raise WordBankError("Word bank is empty")

    return entries


def _parse_word_bank_entry(index: int, entry: object) -> WordBankEntry:
    if not isinstance(entry, dict):
        raise WordBankError(f"Word bank entry {index} must be an object")

    word = entry.get("word")
    hint = entry.get("hint")
    if not isinstance(word, str) or not isinstance(hint, str):
        raise WordBankError(f"Word bank entry {index} must contain word and hint strings")

    normalized_word = word.strip().lower()
    normalized_hint = normalize_imposter_hint(hint).strip().lower()
    if not normalized_word:
        raise WordBankError(f"Word bank entry {index} has an empty word")
    if not normalized_hint:
        raise WordBankError(f"Word bank entry {index} has an empty hint")
    if normalized_word == normalized_hint:
        raise WordBankError(f"Word bank entry {index} uses the word as its hint: {word}")
    if not re.fullmatch(r"[a-z][a-z'-]*", normalized_hint):
        raise WordBankError(f"Word bank entry {index} hint must be one word: {hint}")

    return WordBankEntry(word=normalized_word, hint=normalized_hint)
