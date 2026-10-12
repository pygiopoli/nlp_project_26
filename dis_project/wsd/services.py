"""Run the registered algorithms and collect their results for the templates.

This module does not depend on Django: it returns plain dataclasses that the
views pass to the templates as they are.
"""
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass

from algorithms.base import AlgorithmSpec, WSDResult
from algorithms.nltk_utils import find_target, lemma_forms, penn_to_wn, tokenize_and_tag
from algorithms.registry import ALGORITHMS

logger = logging.getLogger(__name__)

# A run of letters; the capturing group makes re.split keep the words.
WORD_PATTERN = re.compile(r"([A-Za-z]+)")


@dataclass(frozen=True)
class TaggedToken:
    """A token of the sentence with its POS tags."""

    token: str
    penn_tag: str
    wordnet_pos: str | None
    is_target: bool = False


@dataclass(frozen=True)
class Segment:
    """A piece of text, highlighted when it is part of the evidence."""

    text: str
    highlighted: bool = False


@dataclass(frozen=True)
class AlgorithmRun:
    """The outcome of one algorithm: a result, or an error message."""

    key: str
    label: str
    result: WSDResult | None    # None when the algorithm failed
    error: str | None
    elapsed_ms: float
    # The sentence and the chosen sense's definition, split for highlighting.
    sentence_segments: tuple[Segment, ...] = ()
    definition_segments: tuple[Segment, ...] = ()


@dataclass(frozen=True)
class DisambiguationReport:
    """Everything the result page needs."""

    sentence: str
    target: str
    tagged_tokens: tuple[TaggedToken, ...]
    runs: tuple[AlgorithmRun, ...]
    # True/False when at least two algorithms chose a sense, None otherwise.
    agreement: bool | None


def disambiguate(
    sentence: str,
    target: str,
    algorithm_keys: list[str],
    registry: Mapping[str, AlgorithmSpec] = ALGORITHMS,
) -> DisambiguationReport:
    """Run the given algorithms on the sentence and collect their outcomes.

    `registry` can be replaced in tests. An unknown key raises KeyError.
    """
    specs = [registry[key] for key in algorithm_keys]
    tagged_tokens = tag_sentence(sentence, target)
    runs = tuple(run_algorithm(spec, sentence, target) for spec in specs)
    return DisambiguationReport(
        sentence=sentence,
        target=target,
        tagged_tokens=tagged_tokens,
        runs=runs,
        agreement=compute_agreement(runs),
    )


def tag_sentence(sentence: str, target: str) -> tuple[TaggedToken, ...]:
    """POS-tag the sentence (Penn Treebank and WordNet tags) and mark the target."""
    tagged = tokenize_and_tag(sentence)
    match = find_target(tagged, target)
    target_index = match[0] if match else None
    return tuple(
        TaggedToken(token, tag, penn_to_wn(tag), index == target_index)
        for index, (token, tag) in enumerate(tagged)
    )


def run_algorithm(spec: AlgorithmSpec, sentence: str, target: str) -> AlgorithmRun:
    """Run one algorithm, turning any exception into an error message.

    Catching every exception is deliberate: in comparison mode one failing
    algorithm must not break the whole page. The traceback goes to the log.
    """
    start = time.perf_counter()
    try:
        result, error = spec.run(sentence, target), None
    except Exception as exc:
        logger.exception("Algorithm %r failed", spec.key)
        result, error = None, f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - start) * 1000

    sentence_segments: tuple[Segment, ...] = ()
    definition_segments: tuple[Segment, ...] = ()
    if result and result.best:
        matched_words = result.best.matched_words
        sentence_segments = highlight(sentence, matched_words)
        definition_segments = highlight(result.best.sense.definition, matched_words)
    return AlgorithmRun(
        spec.key, spec.label, result, error, elapsed_ms, sentence_segments, definition_segments
    )


def highlight(text: str, matched_words: tuple[str, ...]) -> tuple[Segment, ...]:
    """Split `text` into segments, highlighting the words that match the evidence.

    A word matches when one of its possible lemmas (lowercased) is among the
    matched words, so "prescribed" matches "prescribe". Spaces and punctuation
    are kept: joining the segments gives back the original text.
    """
    matched = {word.lower() for word in matched_words}
    segments = []
    for piece in WORD_PATTERN.split(text):
        if not piece:
            continue
        is_word = WORD_PATTERN.fullmatch(piece) is not None
        segments.append(Segment(piece, is_word and bool(lemma_forms(piece) & matched)))
    return tuple(segments)


def compute_agreement(runs: tuple[AlgorithmRun, ...]) -> bool | None:
    """Whether all the algorithms that chose a sense chose the same one."""
    chosen = [run.result.best.sense.id for run in runs if run.result and run.result.best]
    if len(chosen) < 2:
        return None
    return len(set(chosen)) == 1
