import json

from app.services.word_bank import WORD_BANK_PATH, list_word_bank_entries, normalize_imposter_hint, select_random_word


def test_committed_word_bank_has_expected_shape() -> None:
    raw_entries = json.loads(WORD_BANK_PATH.read_text(encoding="utf-8"))
    entries = list_word_bank_entries()
    words = [entry.word for entry in entries]

    assert len(entries) == 500
    assert all(set(entry) == {"word", "hint"} for entry in raw_entries)
    assert len(set(words)) == len(words)
    assert all(entry.word for entry in entries)
    assert all(entry.hint for entry in entries)
    assert all(" " not in entry.hint for entry in entries)
    assert all(entry.word != entry.hint for entry in entries)


def test_select_random_word_returns_word_and_hint() -> None:
    entries = list_word_bank_entries()

    word, hint = select_random_word(choice_func=lambda choices: choices[0])

    assert word == entries[0].word
    assert hint == entries[0].hint


def test_normalize_imposter_hint_returns_first_word() -> None:
    assert normalize_imposter_hint("Space, signals, or orbit") == "Space"
    assert normalize_imposter_hint("   ") == "thing"
