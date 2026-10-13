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
    highlighted: bool = False   # part of the evidence (matched word)
    is_target: bool = False     # the word being disambiguated


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
    # JSON-serializable data for the score chart; None when there are no candidates.
    chart_data: dict[str, list] | None = None


@dataclass(frozen=True)
class SenseGroup:
    """A chosen sense and the algorithms that chose it."""

    sense_id: str
    definition: str
    runs: tuple[AlgorithmRun, ...]


@dataclass(frozen=True)
class DisambiguationReport:
    """Everything the result page needs."""

    sentence: str
    target: str
    tagged_tokens: tuple[TaggedToken, ...]
    runs: tuple[AlgorithmRun, ...]
    # The chosen senses, most chosen first.
    sense_groups: tuple[SenseGroup, ...]
    # Runs that chose no sense (error or no candidates): excluded from the agreement.
    runs_without_choice: tuple[AlgorithmRun, ...]

    @property
    def agreement(self) -> bool | None:
        """True/False when at least two algorithms chose a sense, None otherwise."""
        chosen_count = sum(len(group.runs) for group in self.sense_groups)
        if chosen_count < 2:
            return None
        return len(self.sense_groups) == 1


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
        sense_groups=group_chosen_senses(runs),
        runs_without_choice=tuple(run for run in runs if not chose_a_sense(run)),
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
        sentence_segments = highlight(sentence, matched_words, target)
        definition_segments = highlight(result.best.sense.definition, matched_words)
    return AlgorithmRun(
        spec.key,
        spec.label,
        result,
        error,
        elapsed_ms,
        sentence_segments,
        definition_segments,
        build_chart_data(result),
    )


def build_chart_data(result: WSDResult | None) -> dict[str, list] | None:
    """Sense ids and scores in ranking order (the chosen sense comes first)."""
    if result is None or not result.ranking:
        return None
    return {
        "labels": [candidate.sense.id for candidate in result.ranking],
        "scores": [candidate.score for candidate in result.ranking],
    }


def highlight(
    text: str, matched_words: tuple[str, ...], target: str | None = None
) -> tuple[Segment, ...]:
    """Split `text` into segments, marking matched words and the target word.

    A word matches when one of its possible lemmas (lowercased) is among the
    matched words, so "prescribed" matches "prescribe"; the target is found
    the same way, so "drugs" is marked for the target "drug". Spaces and
    punctuation are kept: joining the segments gives back the original text.
    """
    matched = {word.lower() for word in matched_words}
    target_forms = lemma_forms(target) if target else set()
    segments = []
    for piece in WORD_PATTERN.split(text):
        if not piece:
            continue
        if WORD_PATTERN.fullmatch(piece) is None:   # spaces and punctuation
            segments.append(Segment(piece))
            continue
        forms = lemma_forms(piece)
        segments.append(Segment(piece, bool(forms & matched), bool(forms & target_forms)))
    return tuple(segments)


def chose_a_sense(run: AlgorithmRun) -> bool:
    return run.result is not None and run.result.best is not None


def group_chosen_senses(runs: tuple[AlgorithmRun, ...]) -> tuple[SenseGroup, ...]:
    """Group the runs by chosen sense, most chosen first (ties keep the run order)."""
    runs_by_sense: dict[str, list[AlgorithmRun]] = {}
    for run in runs:
        if chose_a_sense(run):
            runs_by_sense.setdefault(run.result.best.sense.id, []).append(run)
    groups = [
        SenseGroup(sense_id, group_runs[0].result.best.sense.definition, tuple(group_runs))
        for sense_id, group_runs in runs_by_sense.items()
    ]
    return tuple(sorted(groups, key=lambda group: -len(group.runs)))
