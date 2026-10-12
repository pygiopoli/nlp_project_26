from unittest import TestCase

import nltk
from nltk.wsd import lesk as nltk_lesk

from algorithms.lesk import LESK_KEY, LESK_POS_KEY, lesk
from algorithms.nltk_utils import find_target

DRUG_SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"

# (sentence, target) pairs used to check that the winner matches nltk.wsd.lesk.
EQUIVALENCE_CASES = [
    (DRUG_SENTENCE, "drug"),
    ("I went to the bank to deposit my money", "bank"),
    ("The patient has a cold", "cold"),
    ("The left ventricle of the heart pumps blood to the body", "ventricle"),
    ("She plays the bass in a jazz band", "bass"),
]


class LeskEquivalenceTests(TestCase):
    """The winner must be the synset returned by nltk.wsd.lesk."""

    def test_same_winner_without_pos(self) -> None:
        for sentence, target in EQUIVALENCE_CASES:
            with self.subTest(target=target):
                expected = nltk_lesk(nltk.word_tokenize(sentence), target)
                self.assertEqual(lesk(sentence, target).best.sense.id, expected.name())

    def test_same_winner_with_pos(self) -> None:
        for sentence, target in EQUIVALENCE_CASES:
            with self.subTest(target=target):
                tokens = nltk.word_tokenize(sentence)
                _index, _token, pos = find_target(nltk.pos_tag(tokens), target)
                expected = nltk_lesk(tokens, target, pos=pos)
                result = lesk(sentence, target, use_pos=True)
                self.assertEqual(result.best.sense.id, expected.name())


class LeskResultTests(TestCase):
    def test_drug_ranking_and_overlap(self) -> None:
        result = lesk(DRUG_SENTENCE, "drug")
        self.assertEqual(result.algorithm, LESK_KEY)
        self.assertIsNone(result.pos_filter)
        overlaps = {c.sense.id: c.matched_words for c in result.ranking}
        self.assertEqual(overlaps, {
            "drug.v.01": ("to",),
            "drug.v.02": ("drugs",),
            "drug.n.01": (),
        })
        self.assertEqual(result.best.sense.id, "drug.v.01")
        self.assertTrue(result.is_fallback)  # drug.v.01 and drug.v.02 are tied

    def test_drug_with_pos_filter(self) -> None:
        result = lesk(DRUG_SENTENCE, "drug", use_pos=True)
        self.assertEqual(result.algorithm, LESK_POS_KEY)
        self.assertEqual(result.pos_filter, "n")
        self.assertEqual([c.sense.id for c in result.ranking], ["drug.n.01"])

    def test_word_without_synsets_gives_empty_ranking(self) -> None:
        result = lesk("He said xyzzy loudly", "xyzzy")
        self.assertEqual(result.ranking, ())
        self.assertIsNone(result.best)
        self.assertFalse(result.is_fallback)

    def test_pos_filter_requires_target_in_sentence(self) -> None:
        with self.assertRaises(ValueError):
            lesk("The cat sat on the mat", "dog", use_pos=True)
