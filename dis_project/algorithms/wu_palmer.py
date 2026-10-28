"""Wu-Palmer similarity, optionally combined with phonetic similarity (Tasks 3 and 6).

Ported from scripts/p5_6_phonetic/task_6.ipynb (wup_phonetic), which extends
the Wu-Palmer method of pywsd (max_similarity with option="wup"): each sense
of the target gets, for every context word, its best similarity with one of
the senses of that word, and these values are summed. With use_phonetic=True
the similarity is the convex combination ALPHA * Wu-Palmer + (1 - ALPHA) *
phonetic similarity.

The notebook runs on pywsd's WordNet (OEWN 2024); here NLTK's WordNet is used,
as in the other algorithms, so that the chosen senses can be compared. The
bodies of soundex (scripts/p5_6_phonetic/utils.py), phonetic_similarity and
sense_phonetic are unchanged.
"""
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.metrics.distance import edit_distance

from algorithms.base import ScoredSense, WSDResult
from algorithms.nltk_utils import find_target, rank, sense_from_synset, tokenize_and_tag

WU_PALMER_KEY = "wu_palmer"
WU_PALMER_PHONETIC_KEY = "wu_palmer_phonetic"

# Weight of Wu-Palmer in the phonetic variant (Task 6: 0.5 each).
ALPHA = 0.5


def soundex(word: str, size: int = 4) -> str:
    """Return the Soundex code of a word: first letter + digits, padded with zeros."""
    # Digit for each letter A..Z (0 = ignored: vowels, H, W, Y)
    codes = "01230120022455012623010202"
    code = ""
    for letter in word.upper():
        if not "A" <= letter <= "Z":
            continue
        if not code:
            code = letter                                  # keep the first letter
        else:
            digit = codes[ord(letter) - ord("A")]
            if digit != "0" and code[-1] != digit:         # skip repeated digits
                code += digit
        if len(code) == size:
            break
    return code.ljust(size, "0")


def phonetic_similarity(w1: str, w2: str) -> float:
    """Phonetic similarity between two words, from 0 (completely different) to 1 (same sound).
    Soundex codes always have 4 characters, so the Levenshtein distance goes from 0 to 4."""
    distance = edit_distance(soundex(w1), soundex(w2))
    return 1 - distance / 4


def sense_phonetic(s1: Synset, s2: Synset) -> float:
    """Phonetic similarity between two senses: the best value among all the pairs of their lemmas."""
    return max(phonetic_similarity(a, b) for a in s1.lemma_names() for b in s2.lemma_names())


def wup(s1: Synset, s2: Synset) -> float:
    """Wu-Palmer similarity as in pywsd: 0 when the senses have no common ancestor
    (e.g. a noun and a verb). NLTK's default simulated root would give a value."""
    return s1.wup_similarity(s2, simulate_root=False) or 0.0


def wu_palmer(sentence: str, target: str, use_phonetic: bool = False) -> WSDResult:
    """Disambiguate `target` in `sentence` with Wu-Palmer (+ phonetic) similarity.

    The target must appear in the sentence (ValueError otherwise). The details
    of each sense hold the contribution of each context word to its score.
    """
    tagged = tokenize_and_tag(sentence)
    match = find_target(tagged, target)
    if match is None:
        raise ValueError(f"{target!r} does not appear in the sentence")
    idx, _tok, _pos = match
    alpha = ALPHA if use_phonetic else 1.0
    context = [tok.lower() for i, (tok, _tag) in enumerate(tagged) if i != idx]

    scored = []
    for s in wn.synsets(target):
        # Best similarity of each distinct context word (in sentence order).
        best = {
            word: max((alpha * wup(s, k) + (1 - alpha) * sense_phonetic(s, k) for k in wn.synsets(word)),
                      default=0.0)
            for word in dict.fromkeys(context)
        }
        score = sum(best[word] for word in context)   # repeated words count every time, as in the notebook
        details = {word: round(value, 3) for word, value in sorted(best.items(), key=lambda x: -x[1]) if value > 0}
        scored.append(ScoredSense(sense_from_synset(s), score, details=details))

    ranking, is_fallback = rank(scored)
    return WSDResult(
        algorithm=WU_PALMER_PHONETIC_KEY if use_phonetic else WU_PALMER_KEY,
        target=target,
        ranking=ranking,
        is_fallback=is_fallback,
    )
