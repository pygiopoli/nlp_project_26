"""The registry of the available algorithms.

To add an algorithm, write its module and add its AlgorithmSpec to
REGISTERED_SPECS: the form, the comparison mode and the result pages pick it
up from ALGORITHMS.
"""
from collections.abc import Iterable
from functools import partial

from algorithms.base import AlgorithmSpec
from algorithms.extended_lesk import EXTENDED_LESK_KEY, EXTENDED_LESK_POS_KEY, extended_lesk
from algorithms.lesk import LESK_KEY, LESK_POS_KEY, lesk
from algorithms.sapbert import SAPBERT_KEY, sapbert
from algorithms.wu_palmer import WU_PALMER_KEY, WU_PALMER_PHONETIC_KEY, wu_palmer

# The order of this list is the order shown in the UI.
REGISTERED_SPECS: list[AlgorithmSpec] = [
    AlgorithmSpec(
        key=LESK_KEY,
        label="Standard Lesk",
        description="Overlap between the sentence and each sense definition (nltk.wsd.lesk).",
        run=partial(lesk, use_pos=False),
    ),
    AlgorithmSpec(
        key=LESK_POS_KEY,
        label="Standard Lesk + POS",
        description="Standard Lesk restricted to the senses with the POS tagged in the sentence.",
        run=partial(lesk, use_pos=True),
    ),
    AlgorithmSpec(
        key=EXTENDED_LESK_KEY,
        label="Extended Lesk",
        description=(
            "Overlap between the context lemmas and an extended signature: definition, "
            "examples and related synsets (hypernyms, hyponyms, meronyms, holonyms...)."
        ),
        run=partial(extended_lesk, use_pos=False),
    ),
    AlgorithmSpec(
        key=EXTENDED_LESK_POS_KEY,
        label="Extended Lesk + POS",
        description="Extended Lesk restricted to the senses with the POS tagged in the sentence.",
        run=partial(extended_lesk, use_pos=True),
    ),
    AlgorithmSpec(
        key=WU_PALMER_KEY,
        label="Wu-Palmer",
        description=(
            "Sum, over the context words, of the best Wu-Palmer similarity between the sense "
            "and the senses of the word (pywsd max_similarity)."
        ),
        run=partial(wu_palmer, use_phonetic=False),
    ),
    AlgorithmSpec(
        key=WU_PALMER_PHONETIC_KEY,
        label="Wu-Palmer + phonetic",
        description="Wu-Palmer with 0.5 · Wu-Palmer + 0.5 · phonetic similarity (Soundex + Levenshtein).",
        run=partial(wu_palmer, use_phonetic=True),
    ),
    AlgorithmSpec(
        key=SAPBERT_KEY,
        label="SapBERT (UMLS senses)",
        description=(
            "Cosine similarity between the SapBERT embeddings of the context and of each "
            "UMLS sense of the MSH-WSD word (names + definition)."
        ),
        run=sapbert,
    ),
]


def build_registry(specs: Iterable[AlgorithmSpec]) -> dict[str, AlgorithmSpec]:
    """Map each algorithm key to its spec, rejecting duplicate keys."""
    registry: dict[str, AlgorithmSpec] = {}
    for spec in specs:
        if spec.key in registry:
            raise ValueError(f"Duplicate algorithm key: {spec.key!r}")
        registry[spec.key] = spec
    return registry


ALGORITHMS: dict[str, AlgorithmSpec] = build_registry(REGISTERED_SPECS)


def get_algorithm(key: str) -> AlgorithmSpec:
    """Return the spec registered under `key`."""
    try:
        return ALGORITHMS[key]
    except KeyError:
        raise KeyError(f"Unknown algorithm: {key!r}") from None
