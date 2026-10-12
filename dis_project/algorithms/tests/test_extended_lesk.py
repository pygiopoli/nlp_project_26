from unittest import TestCase

from algorithms.extended_lesk import EXTENDED_LESK_KEY, EXTENDED_LESK_POS_KEY, extended_lesk

DRUG_SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"
BANK_SENTENCE = "I went to the bank to deposit my money"


class ExtendedLeskTests(TestCase):
    """Expected values are the outputs of scripts/p2/drug_disamb.ipynb."""

    def test_drug_without_pos(self) -> None:
        result = extended_lesk(DRUG_SENTENCE, "drug")
        self.assertEqual(result.algorithm, EXTENDED_LESK_KEY)
        self.assertIsNone(result.pos_filter)
        scores = {c.sense.id: (c.score, c.matched_words) for c in result.ranking}
        self.assertEqual(scores, {
            "drug.n.01": (1, ("two",)),
            "drug.v.01": (0, ()),
            "drug.v.02": (0, ()),
        })
        self.assertEqual(result.best.sense.id, "drug.n.01")
        self.assertFalse(result.is_fallback)

    def test_drug_with_pos(self) -> None:
        result = extended_lesk(DRUG_SENTENCE, "drugs", use_pos=True)
        self.assertEqual(result.algorithm, EXTENDED_LESK_POS_KEY)
        self.assertEqual(result.pos_filter, "n")
        self.assertEqual([c.sense.id for c in result.ranking], ["drug.n.01"])

    def test_bank_with_pos(self) -> None:
        result = extended_lesk(BANK_SENTENCE, "bank", use_pos=True)
        self.assertEqual(result.best.sense.id, "depository_financial_institution.n.01")
        self.assertEqual(result.best.matched_words, ("deposit", "money"))
        self.assertFalse(result.is_fallback)

    def test_word_without_synsets_gives_empty_ranking(self) -> None:
        result = extended_lesk("He said xyzzy loudly", "xyzzy")
        self.assertEqual(result.ranking, ())
        self.assertIsNone(result.best)

    def test_target_missing_from_sentence_raises(self) -> None:
        with self.assertRaises(ValueError):
            extended_lesk("The cat sat on the mat", "dog")
