from unittest import TestCase

from nltk.corpus import wordnet as wn

from algorithms.wu_palmer import (
    WU_PALMER_KEY, WU_PALMER_PHONETIC_KEY, phonetic_similarity, soundex, wu_palmer, wup,
)

DRUG_SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"


class PhoneticTests(TestCase):
    """Expected values are the outputs of scripts/p5_6_phonetic/task_6.ipynb."""

    def test_soundex_codes(self) -> None:
        codes = {word: soundex(word) for word in ("drug", "drugs", "dose", "two", "today")}
        self.assertEqual(codes, {"drug": "D620", "drugs": "D620", "dose": "D200", "two": "T000", "today": "T300"})

    def test_phonetic_similarity(self) -> None:
        self.assertEqual(phonetic_similarity("drug", "drugs"), 1.0)
        self.assertEqual(phonetic_similarity("two", "to"), 1.0)
        self.assertEqual(phonetic_similarity("drug", "dose"), 0.5)


class WupTests(TestCase):
    def test_same_sense(self) -> None:
        self.assertEqual(wup(wn.synset("drug.n.01"), wn.synset("drug.n.01")), 1.0)

    def test_no_common_ancestor_gives_zero(self) -> None:
        self.assertEqual(wup(wn.synset("drug.v.01"), wn.synset("clinic.n.01")), 0.0)


class WuPalmerTests(TestCase):
    def test_drug_without_phonetic(self) -> None:
        result = wu_palmer(DRUG_SENTENCE, "drug")
        self.assertEqual(result.algorithm, WU_PALMER_KEY)
        self.assertIsNone(result.pos_filter)
        self.assertEqual(result.best.sense.id, "drug.n.01")
        self.assertFalse(result.is_fallback)

    def test_drug_with_phonetic(self) -> None:
        # As in the notebook, the phonetic part makes "administer a drug to" win.
        result = wu_palmer(DRUG_SENTENCE, "drug", use_phonetic=True)
        self.assertEqual(result.algorithm, WU_PALMER_PHONETIC_KEY)
        self.assertEqual(result.best.sense.id, "drug.v.01")

    def test_details_are_the_contributions_of_the_context_words(self) -> None:
        best = wu_palmer(DRUG_SENTENCE, "drug").best
        self.assertNotIn("drugs", best.details)
        self.assertAlmostEqual(sum(best.details.values()), best.score, places=2)
        contributions = list(best.details.values())
        self.assertEqual(contributions, sorted(contributions, reverse=True))

    def test_word_without_synsets_gives_empty_ranking(self) -> None:
        result = wu_palmer("He said xyzzy loudly", "xyzzy", use_phonetic=True)
        self.assertEqual(result.ranking, ())
        self.assertIsNone(result.best)

    def test_target_missing_from_sentence_raises(self) -> None:
        with self.assertRaises(ValueError):
            wu_palmer("The cat sat on the mat", "dog")
