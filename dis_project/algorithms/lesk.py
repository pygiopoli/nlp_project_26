"""Standard Lesk, as implemented by nltk.wsd.lesk, with the full ranking.

nltk.wsd.lesk returns only the winning synset. This module reproduces its
scoring (overlap between the sentence tokens and the words of each definition)
so that every candidate's score and overlapping words can be shown.

Like NLTK, it is case-sensitive, keeps stopwords and does not remove the
target word from the context: these quirks are kept on purpose.
"""
import nltk
from nltk.corpus import wordnet as wn

from algorithms.base import ScoredSense, WSDResult
from algorithms.nltk_utils import find_target, rank, sense_from_synset

LESK_KEY = "lesk"
LESK_POS_KEY = "lesk_pos"


def lesk(sentence: str, target: str, use_pos: bool = False) -> WSDResult:
    """Disambiguate `target` in `sentence` with standard Lesk.

    With use_pos=True, the candidates are restricted to the POS that the
    tagger assigns to the target; the target must then appear in the sentence
    (ValueError otherwise).
    """
    tokens = nltk.word_tokenize(sentence)
    pos = _target_pos(tokens, target) if use_pos else None

    synsets = wn.synsets(target)
    if pos:
        # Same filter as nltk.wsd.lesk: adjective satellites ("s") are not kept for "a".
        synsets = [synset for synset in synsets if synset.pos() == pos]

    context = set(tokens)
    scored = []
    for synset in synsets:
        overlap = sorted(context & set(synset.definition().split()))
        scored.append(ScoredSense(sense_from_synset(synset), len(overlap), tuple(overlap)))

    ranking, is_fallback = rank(scored)
    return WSDResult(
        algorithm=LESK_POS_KEY if use_pos else LESK_KEY,
        target=target,
        ranking=ranking,
        pos_filter=pos,
        is_fallback=is_fallback,
    )


def _target_pos(tokens: list[str], target: str) -> str | None:
    """The WordNet POS the tagger assigns to the target in the sentence."""
    match = find_target(nltk.pos_tag(tokens), target)
    if match is None:
        raise ValueError(f"{target!r} does not appear in the sentence")
    _index, _token, pos = match
    return pos
