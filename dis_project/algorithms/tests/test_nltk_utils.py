from unittest import TestCase

from algorithms.base import ScoredSense, Sense
from algorithms.nltk_utils import find_target, rank, tokenize_and_tag, warm_up

DRUG_SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"


def make_scored(sense_id: str, score: float) -> ScoredSense:
    return ScoredSense(Sense(sense_id, "n", ""), score)


class FindTargetTests(TestCase):
    def test_matches_inflected_form_from_lemma(self) -> None:
        self.assertEqual(find_target(tokenize_and_tag(DRUG_SENTENCE), "drug"), (6, "drugs", "n"))

    def test_matches_target_given_in_inflected_form(self) -> None:
        self.assertEqual(find_target(tokenize_and_tag(DRUG_SENTENCE), "drugs"), (6, "drugs", "n"))

    def test_is_case_insensitive(self) -> None:
        self.assertEqual(find_target(tokenize_and_tag(DRUG_SENTENCE), "Drug"), (6, "drugs", "n"))
        tagged = tokenize_and_tag("The patient was given an HIV test")
        self.assertEqual(find_target(tagged, "HIV")[1], "HIV")

    def test_returns_none_when_missing(self) -> None:
        self.assertIsNone(find_target(tokenize_and_tag("The cat sat on the mat"), "dog"))


class RankTests(TestCase):
    def test_sorts_by_descending_score_keeping_order_on_ties(self) -> None:
        ranking, _ = rank([make_scored("a", 1), make_scored("b", 2), make_scored("c", 1)])
        self.assertEqual([c.sense.id for c in ranking], ["b", "a", "c"])

    def test_single_best_is_not_a_fallback(self) -> None:
        _, is_fallback = rank([make_scored("a", 1), make_scored("b", 2)])
        self.assertFalse(is_fallback)

    def test_tie_at_the_top_is_a_fallback(self) -> None:
        _, is_fallback = rank([make_scored("a", 2), make_scored("b", 2)])
        self.assertTrue(is_fallback)

    def test_zero_best_score_is_a_fallback(self) -> None:
        _, is_fallback = rank([make_scored("a", 0)])
        self.assertTrue(is_fallback)

    def test_empty_input(self) -> None:
        self.assertEqual(rank([]), ((), False))


class WarmUpTests(TestCase):
    def test_runs_without_errors(self) -> None:
        warm_up()
