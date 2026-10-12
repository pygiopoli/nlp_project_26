"""Extended (adapted) Lesk, ported from scripts/p2/drug_disamb.ipynb.

The signature of each sense is built from its definition and examples plus
the lemmas, definitions and examples of its related synsets; the context is
the set of lemmas of the other words in the sentence (POS-aware). The score
is the number of shared lemmas.

The bodies of related_synsets, normalize, extended_gloss and context_lemmas
are unchanged from the notebook; only the shared resources (lemmatizer,
stopwords, penn_to_wn) now come from algorithms.nltk_utils.
"""
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.tokenize import word_tokenize

from algorithms.base import ScoredSense, WSDResult
from algorithms.nltk_utils import (
    find_target, lemmatizer, penn_to_wn, rank, sense_from_synset, stop_words, tokenize_and_tag,
)

EXTENDED_LESK_KEY = "extended_lesk"
EXTENDED_LESK_POS_KEY = "extended_lesk_pos"

# How many levels of hyponyms are added to the signature.
HYPONYM_DEPTH = 2


def related_synsets(s: Synset, depth: int = HYPONYM_DEPTH) -> list[Synset]:
    """Related synsets; hyponyms are followed `depth` levels down."""
    rel = []
    rel += s.hypernyms()
    level = s.hyponyms()
    for _ in range(depth):
        rel += level
        level = [h for x in level for h in x.hyponyms()]
    rel += s.part_meronyms() + s.member_meronyms() + s.substance_meronyms()
    rel += s.part_holonyms() + s.member_holonyms() + s.substance_holonyms()
    rel += s.similar_tos() + s.also_sees() + s.attributes()
    rel += s.entailments() + s.verb_groups()
    return rel


def normalize(text: str) -> list[str]:
    """Lowercase, tokenize, drop stopwords/punctuation, lemmatize."""
    out = []
    for w in word_tokenize(text.lower()):
        if w.isalpha() and w not in stop_words():
            out.append(lemmatizer().lemmatize(w))
    return out


def extended_gloss(s: Synset) -> list[str]:
    words = normalize(s.definition())
    for ex in s.examples():
        words += normalize(ex)
    for r in related_synsets(s):
        words += [l.lower() for name in r.lemma_names() for l in name.split("_")]
        words += normalize(r.definition())
        for ex in r.examples():
            words += normalize(ex)
    return words


def context_lemmas(tagged: list[tuple[str, str]], idx: int) -> set[str]:
    """Lemmas of the context words, using their own POS tag (prescribed -> prescribe)."""
    out = set()
    for i, (tok, tag) in enumerate(tagged):
        if i == idx or tok.lower() in stop_words() or not tok.isalpha():
            continue
        out.add(lemmatizer().lemmatize(tok.lower(), penn_to_wn(tag) or wn.NOUN))
    return out


def extended_lesk(sentence: str, target: str, use_pos: bool = False) -> WSDResult:
    """Disambiguate `target` in `sentence` with extended Lesk.

    The target must appear in the sentence (ValueError otherwise). With
    use_pos=True, the candidates are restricted to the POS that the tagger
    assigns to the target.
    """
    tagged = tokenize_and_tag(sentence)
    match = find_target(tagged, target)
    if match is None:
        raise ValueError(f"{target!r} does not appear in the sentence")
    idx, tok, target_pos = match
    pos = target_pos if use_pos else None

    lemma = lemmatizer().lemmatize(tok.lower(), pos or wn.NOUN)
    context = context_lemmas(tagged, idx)
    scored = []
    for s in wn.synsets(lemma, pos=pos):
        overlap = sorted(context & set(extended_gloss(s)))
        scored.append(ScoredSense(sense_from_synset(s), len(overlap), tuple(overlap)))

    ranking, is_fallback = rank(scored)
    return WSDResult(
        algorithm=EXTENDED_LESK_POS_KEY if use_pos else EXTENDED_LESK_KEY,
        target=target,
        ranking=ranking,
        pos_filter=pos,
        is_fallback=is_fallback,
    )
