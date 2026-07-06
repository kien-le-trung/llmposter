from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BAD_HINTS = {
    "abstraction",
    "act",
    "attribute",
    "causal_agent",
    "communication",
    "condition",
    "content",
    "deed",
    "entity",
    "event",
    "group",
    "discipline",
    "hypostasis",
    "location",
    "object",
    "operation",
    "part",
    "physical_entity",
    "period",
    "preoccupation",
    "property",
    "psychological_feature",
    "relation",
    "state",
    "subject",
    "substance",
    "thing",
    "unit",
    "whole",
}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "backend" / "app" / "data" / "word_bank.json"
WORD_PATTERN = re.compile(r"^[a-z]{3,16}$")
BAD_WORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "because",
    "before",
    "being",
    "between",
    "both",
    "could",
    "did",
    "does",
    "doing",
    "done",
    "down",
    "each",
    "even",
    "ever",
    "every",
    "first",
    "from",
    "get",
    "gets",
    "getting",
    "give",
    "goes",
    "going",
    "gone",
    "good",
    "got",
    "had",
    "has",
    "have",
    "having",
    "here",
    "into",
    "just",
    "like",
    "make",
    "many",
    "making",
    "may",
    "means",
    "men",
    "more",
    "most",
    "much",
    "must",
    "need",
    "now",
    "once",
    "only",
    "other",
    "out",
    "over",
    "own",
    "same",
    "see",
    "shall",
    "should",
    "someone",
    "some",
    "such",
    "take",
    "than",
    "that",
    "then",
    "there",
    "these",
    "thing",
    "things",
    "this",
    "those",
    "through",
    "under",
    "until",
    "very",
    "want",
    "was",
    "well",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "would",
    "years",
    "yes",
    "why",
}
BAD_WORDS.update(
    {
        "april",
        "august",
        "david",
        "december",
        "february",
        "james",
        "january",
        "john",
        "july",
        "june",
        "lives",
        "march",
        "michael",
        "november",
        "october",
        "playing",
        "saying",
        "september",
        "using",
        "washington",
        "york",
    }
)
BAD_WORD_SUFFIXES = (
    "ing",
    "ism",
    "ology",
    "logy",
    "ness",
    "ity",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the committed word bank seed.")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--candidate-count", type=int, default=75000)
    parser.add_argument("--lexicon", default="ewn:2020")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-file",
        type=Path,
        default=None,
        help="Optional newline-delimited source word list, such as tube42 data/en.add.",
    )
    args = parser.parse_args()

    import wn

    _ensure_lexicon(wn, args.lexicon)
    wordnet = wn.Wordnet(lexicon=args.lexicon)
    source_words = _read_source_words(args.source_file) if args.source_file else None
    entries = _build_entries(wordnet, args.limit, args.candidate_count, source_words)

    if len(entries) < args.limit:
        raise SystemExit(f"Only generated {len(entries)} entries; expected {args.limit}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(entries)} entries to {args.output}")


def _ensure_lexicon(wn_module: Any, lexicon: str) -> None:
    if wn_module.lexicons(lexicon=lexicon):
        return

    wn_module.download(lexicon)


def _read_source_words(source_file: Path) -> set[str]:
    words: set[str] = set()
    for line in source_file.read_text(encoding="utf-8").splitlines():
        word = _normalize_word(line)
        if word is not None:
            words.add(word)

    return words


def _build_entries(
    wordnet: Any,
    limit: int,
    candidate_count: int,
    source_words: set[str] | None = None,
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen_words: set[str] = set()

    for lemma in _candidate_words(wordnet, candidate_count, source_words):
        if lemma is None or lemma in seen_words:
            continue

        synsets = wordnet.synsets(lemma, pos="n")
        hint = _select_hint(synsets, lemma)
        if hint is None:
            continue

        seen_words.add(lemma)
        entries.append(
            {
                "word": lemma,
                "hint": hint,
            }
        )
        if len(entries) >= limit:
            return entries

    return entries


def _candidate_words(
    wordnet: Any,
    candidate_count: int,
    source_words: set[str] | None,
) -> Iterable[str]:
    try:
        from wordfreq import top_n_list
    except ImportError:
        top_words: Iterable[str] = []
    else:
        top_words = top_n_list("en", candidate_count)

    yielded: set[str] = set()
    for word in top_words:
        normalized = _normalize_word(word)
        if (
            normalized is not None
            and normalized not in yielded
            and _is_allowed_source_word(normalized, source_words)
            and _has_only_noun_entries(wordnet, normalized)
            and not _looks_plural(wordnet, normalized)
        ):
            yielded.add(normalized)
            yield normalized

    for word in wordnet.words(pos="n"):
        normalized = _normalize_word(word.lemma())
        if (
            normalized is not None
            and normalized not in yielded
            and _is_allowed_source_word(normalized, source_words)
            and not _looks_plural(wordnet, normalized)
        ):
            yielded.add(normalized)
            yield normalized


def _is_allowed_source_word(word: str, source_words: set[str] | None) -> bool:
    return source_words is None or word in source_words


def _has_only_noun_entries(wordnet: Any, word: str) -> bool:
    parts_of_speech = {entry.pos for entry in wordnet.words(word)}
    return parts_of_speech == {"n"}


def _looks_plural(wordnet: Any, word: str) -> bool:
    if not word.endswith("s") or word.endswith("ss") or len(word) <= 4:
        return False

    return bool(wordnet.synsets(word[:-1], pos="n"))


def _select_hint(synsets: Iterable[Any], word: str) -> str | None:
    for synset in synsets:
        for hypernym in _hypernym_candidates(synset):
            hint = _select_lemma(hypernym, word)
            if hint is not None:
                return hint

    return None


def _hypernym_candidates(synset: Any) -> list[Any]:
    direct = list(synset.hypernyms())
    if direct:
        expanded: list[Any] = []
        for hypernym in direct:
            expanded.append(hypernym)
            expanded.extend(hypernym.hypernyms())
        return expanded

    candidates: list[Any] = []
    for path in synset.hypernym_paths():
        candidates.extend(reversed(path[:-1]))
    return candidates


def _select_lemma(synset: Any, word: str) -> str | None:
    for lemma in synset.lemmas():
        normalized = _normalize_hint(lemma)
        if normalized is None:
            continue
        if normalized == word or normalized in BAD_HINTS:
            continue
        if normalized in word or word in normalized:
            continue
        return normalized

    return None


def _normalize_word(value: str) -> str | None:
    normalized = value.strip().lower().replace("_", " ")
    if " " in normalized or "-" in normalized or "'" in normalized:
        return None
    if WORD_PATTERN.fullmatch(normalized) is None:
        return None
    if normalized in BAD_WORDS:
        return None
    if normalized.endswith(BAD_WORD_SUFFIXES):
        return None

    return normalized


def _normalize_hint(value: str) -> str | None:
    normalized = _normalize_word(value)
    if normalized is None:
        return None
    if len(normalized) < 3:
        return None

    return normalized


if __name__ == "__main__":
    main()
