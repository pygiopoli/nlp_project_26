from unittest import TestCase

from algorithms.base import ScoredSense, Sense, WSDResult


def make_scored(sense_id: str, score: float) -> ScoredSense:
    return ScoredSense(Sense(sense_id, "n", f"definition of {sense_id}"), score)


class WSDResultTests(TestCase):
    def test_best_is_none_without_candidates(self) -> None:
        result = WSDResult(algorithm="test", target="xyzzy", ranking=())
        self.assertIsNone(result.best)

    def test_best_is_first_in_ranking(self) -> None:
        first = make_scored("drug.n.01", 2)
        second = make_scored("drug.v.01", 1)
        result = WSDResult(algorithm="test", target="drug", ranking=(first, second))
        self.assertEqual(result.best, first)
