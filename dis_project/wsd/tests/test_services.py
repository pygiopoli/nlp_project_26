from django.test import SimpleTestCase

from algorithms.base import AlgorithmSpec, ScoredSense, Sense, WSDResult
from algorithms.registry import ALGORITHMS
from wsd.services import Segment, build_chart_data, disambiguate, highlight

DRUG_SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"


def failing_run(sentence: str, target: str) -> WSDResult:
    raise RuntimeError("boom")


FAILING_SPEC = AlgorithmSpec(key="failing", label="Failing", description="", run=failing_run)


class DisambiguateTests(SimpleTestCase):
    def test_single_algorithm(self) -> None:
        report = disambiguate(DRUG_SENTENCE, "drug", ["extended_lesk"])
        self.assertEqual(len(report.runs), 1)
        run = report.runs[0]
        self.assertEqual((run.key, run.label), ("extended_lesk", "Extended Lesk"))
        self.assertIsNone(run.error)
        self.assertEqual(run.result.best.sense.id, "drug.n.01")
        self.assertGreaterEqual(run.elapsed_ms, 0)
        self.assertIsNone(report.agreement)

    def test_runs_keep_the_requested_order(self) -> None:
        keys = ["extended_lesk_pos", "lesk"]
        report = disambiguate(DRUG_SENTENCE, "drug", keys)
        self.assertEqual([run.key for run in report.runs], keys)

    def test_disagreement(self) -> None:
        # lesk chooses drug.v.01, extended_lesk chooses drug.n.01
        report = disambiguate(DRUG_SENTENCE, "drug", ["lesk", "extended_lesk"])
        self.assertIs(report.agreement, False)

    def test_agreement(self) -> None:
        report = disambiguate(DRUG_SENTENCE, "drug", ["lesk_pos", "extended_lesk"])
        self.assertIs(report.agreement, True)

    def test_failing_algorithm_does_not_stop_the_others(self) -> None:
        registry = {**ALGORITHMS, FAILING_SPEC.key: FAILING_SPEC}
        with self.assertLogs("wsd.services", level="ERROR"):
            report = disambiguate(DRUG_SENTENCE, "drug", ["failing", "lesk"], registry=registry)
        failed, succeeded = report.runs
        self.assertIsNone(failed.result)
        self.assertEqual(failed.error, "RuntimeError: boom")
        self.assertEqual(succeeded.result.best.sense.id, "drug.v.01")
        self.assertIsNone(report.agreement)  # only one algorithm chose a sense

    def test_pos_tagging_of_the_sentence(self) -> None:
        report = disambiguate("I went to the bank", "bank", ["lesk"])
        tags = [(t.token, t.penn_tag, t.wordnet_pos) for t in report.tagged_tokens]
        self.assertEqual(tags[-1], ("bank", "NN", "n"))
        self.assertEqual(len(tags), 5)

    def test_unknown_algorithm_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            disambiguate(DRUG_SENTENCE, "drug", ["does-not-exist"])

    def test_target_token_is_marked(self) -> None:
        report = disambiguate(DRUG_SENTENCE, "drug", ["lesk"])
        targets = [t.token for t in report.tagged_tokens if t.is_target]
        self.assertEqual(targets, ["drugs"])

    def test_segments_of_the_chosen_sense(self) -> None:
        report = disambiguate("I went to the bank to deposit my money", "bank", ["extended_lesk_pos"])
        run = report.runs[0]
        highlighted = [s.text for s in run.sentence_segments if s.highlighted]
        self.assertEqual(highlighted, ["deposit", "money"])
        self.assertEqual("".join(s.text for s in run.definition_segments),
                         run.result.best.sense.definition)

    def test_failed_run_has_no_segments(self) -> None:
        registry = {FAILING_SPEC.key: FAILING_SPEC}
        with self.assertLogs("wsd.services", level="ERROR"):
            report = disambiguate(DRUG_SENTENCE, "drug", ["failing"], registry=registry)
        self.assertEqual(report.runs[0].sentence_segments, ())


class HighlightTests(SimpleTestCase):
    def test_matches_inflected_forms(self) -> None:
        segments = highlight("I was prescribed drugs.", ("prescribe",))
        self.assertIn(Segment("prescribed", True), segments)

    def test_is_case_insensitive(self) -> None:
        segments = highlight("Money matters", ("money",))
        self.assertEqual(segments[0], Segment("Money", True))

    def test_joined_segments_give_back_the_text(self) -> None:
        text = "a substance that is used as a medicine or narcotic; (e.g. aspirin)"
        segments = highlight(text, ("medicine",))
        self.assertEqual("".join(s.text for s in segments), text)
        self.assertEqual([s.text for s in segments if s.highlighted], ["medicine"])

    def test_marks_the_target_in_any_form(self) -> None:
        segments = highlight("Two drugs and a drug", (), target="drug")
        self.assertEqual([s.text for s in segments if s.is_target], ["drugs", "drug"])

    def test_target_can_also_be_a_matched_word(self) -> None:
        segments = highlight("use recreational drugs", ("drugs",), target="drug")
        self.assertEqual(segments[-1], Segment("drugs", highlighted=True, is_target=True))

    def test_no_matched_words_highlights_nothing(self) -> None:
        self.assertFalse(any(s.highlighted for s in highlight("I went to the bank", ())))


class BuildChartDataTests(SimpleTestCase):
    def test_labels_and_scores_in_ranking_order(self) -> None:
        ranking = (
            ScoredSense(Sense("drug.n.01", "n", ""), 2),
            ScoredSense(Sense("drug.v.01", "v", ""), 0),
        )
        result = WSDResult(algorithm="test", target="drug", ranking=ranking)
        self.assertEqual(build_chart_data(result), {
            "labels": ["drug.n.01", "drug.v.01"],
            "scores": [2, 0],
        })

    def test_no_chart_without_candidates(self) -> None:
        self.assertIsNone(build_chart_data(WSDResult(algorithm="test", target="x", ranking=())))
        self.assertIsNone(build_chart_data(None))


class SenseGroupTests(SimpleTestCase):
    def test_groups_most_chosen_first(self) -> None:
        keys = ["lesk", "lesk_pos", "extended_lesk", "extended_lesk_pos"]
        report = disambiguate(DRUG_SENTENCE, "drug", keys)
        groups = [(g.sense_id, [r.key for r in g.runs]) for g in report.sense_groups]
        self.assertEqual(groups, [
            ("drug.n.01", ["lesk_pos", "extended_lesk", "extended_lesk_pos"]),
            ("drug.v.01", ["lesk"]),
        ])
        self.assertEqual(report.runs_without_choice, ())
        self.assertIs(report.agreement, False)

    def test_runs_without_choice_are_excluded(self) -> None:
        registry = {**ALGORITHMS, FAILING_SPEC.key: FAILING_SPEC}
        with self.assertLogs("wsd.services", level="ERROR"):
            report = disambiguate(
                DRUG_SENTENCE, "drug", ["failing", "lesk_pos", "extended_lesk"], registry=registry
            )
        self.assertEqual([r.key for r in report.runs_without_choice], ["failing"])
        self.assertEqual(len(report.sense_groups), 1)
        self.assertIs(report.agreement, True)

    def test_no_candidates_means_no_groups(self) -> None:
        report = disambiguate("He said xyzzy loudly", "xyzzy", ["lesk", "extended_lesk"])
        self.assertEqual(report.sense_groups, ())
        self.assertEqual(len(report.runs_without_choice), 2)
        self.assertIsNone(report.agreement)
