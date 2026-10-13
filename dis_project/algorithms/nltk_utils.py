"""NLTK helpers shared by the WordNet-based algorithms.

Heavy resources are loaded lazily: NLTK loads WordNet and the tagger on first
use and keeps them in memory; our own derived resources are cached with
functools.cache, so each one is built once per process.
"""
from functools import cache

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.stem import WordNetLemmatizer

from algorithms.base import ScoredSense, Sense

# (index of the target in the tagged sentence, its token, its WordNet POS)
TargetMatch = tuple[int, str, str | None]


@cache
def stop_words() -> frozenset[str]:
    """English stopwords."""
    return frozenset(stopwords.words("english"))


@cache
def lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def warm_up() -> None:
    """Load every lazy resource now, so the first disambiguation is fast.

    Loading WordNet takes a few seconds; afterwards it stays in memory.
    """
    wn.ensure_loaded()
    tokenize_and_tag("Warm up the tokenizer and the tagger.")
    stop_words()
    lemmatizer().lemmatize("warming")


def penn_to_wn(tag: str) -> str | None:
    """Map a Penn Treebank tag to a WordNet POS (None if WordNet has no such POS)."""
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("R"):
        return wn.ADV
    return None


def tokenize_and_tag(sentence: str) -> list[tuple[str, str]]:
    """Tokenize the sentence and return (token, Penn Treebank tag) pairs."""
    return nltk.pos_tag(nltk.word_tokenize(sentence))


def lemma_forms(word: str) -> set[str]:
    """The lowercased word and its possible lemmas (noun, verb, adjective, adverb)."""
    lowered = word.lower()
    forms = {lowered}
    for pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
        forms.add(lemmatizer().lemmatize(lowered, pos))
    return forms


def find_target(tagged: list[tuple[str, str]], target: str) -> TargetMatch | None:
    """Find the first token whose lemma matches a lemma of `target`.

    Matching is case-insensitive and accepts inflected forms, so "drugs",
    "Drug" and "drug" all match. Returns None when the target is not found.
    """
    target_forms = lemma_forms(target)
    for index, (token, tag) in enumerate(tagged):
        pos = penn_to_wn(tag)
        lowered = token.lower()
        token_lemma = lemmatizer().lemmatize(lowered, pos) if pos else lowered
        if token_lemma in target_forms:
            return index, token, pos
    return None


def sense_from_synset(synset: Synset) -> Sense:
    """Convert an NLTK synset into a plain Sense."""
    return Sense(
        id=synset.name(),
        pos=synset.pos(),
        definition=synset.definition(),
        examples=tuple(synset.examples()),
    )


def rank(scored: list[ScoredSense]) -> tuple[tuple[ScoredSense, ...], bool]:
    """Sort candidates by descending score and tell whether the choice is a fallback.

    The sort is stable, so ties keep WordNet's order (most frequent sense
    first), as in nltk.wsd.lesk. The choice is a fallback when the best score
    is 0 or is shared by more than one candidate.
    """
    ranking = tuple(sorted(scored, key=lambda candidate: -candidate.score))
    if not ranking:
        return ranking, False
    top = ranking[0].score
    tied = sum(1 for candidate in ranking if candidate.score == top)
    return ranking, top == 0 or tied > 1
