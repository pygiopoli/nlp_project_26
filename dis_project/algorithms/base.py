"""Common data structures shared by every WSD algorithm.

They hold only plain data (strings and numbers), never NLTK objects, so the
web application can render the result of any algorithm in the same way.
"""
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sense:
    """A candidate sense of the target word."""

    id: str                     # e.g. "drug.n.01" (WordNet) or "C0018827" (UMLS CUI)
    pos: str                    # WordNet POS: "n", "v", "a", "s" or "r"; "" for UMLS concepts
    definition: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredSense:
    """A candidate sense with its score and the optional evidence behind it."""

    sense: Sense
    score: float
    # Words shared by the context and the sense (highlighted in the UI).
    matched_words: tuple[str, ...] = ()
    # Named score components or other values (shown as a key/value table).
    details: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class WSDResult:
    """The output of one algorithm on one sentence."""

    algorithm: str                      # key of the algorithm in the registry
    target: str
    ranking: tuple[ScoredSense, ...]    # by descending score; empty = no candidates
    pos_filter: str | None = None       # WordNet POS used to filter the candidates
    is_fallback: bool = False           # tie or no evidence: most frequent sense chosen

    @property
    def best(self) -> ScoredSense | None:
        """The chosen sense, or None when the target has no candidate senses."""
        return self.ranking[0] if self.ranking else None


@dataclass(frozen=True)
class AlgorithmSpec:
    """Description of a registered algorithm."""

    key: str                                # used in URLs and in the form
    label: str                              # human-readable name shown in the UI
    description: str
    run: Callable[[str, str], WSDResult]    # (sentence, target) -> result
