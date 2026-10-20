from unittest.mock import patch

from django.test import SimpleTestCase
from django.urls import reverse

from algorithms.base import AlgorithmSpec, WSDResult
from algorithms.registry import ALGORITHMS
from wsd.services import DisambiguationReport, disambiguate

DRUG_SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"


def failing_run(sentence: str, target: str) -> WSDResult:
    raise RuntimeError("boom")


def disambiguate_with_failing_algorithm(
    sentence: str, target: str, algorithm_keys: list[str]
) -> DisambiguationReport:
    """Replacement for wsd.views.disambiguate that runs one failing algorithm."""
    spec = AlgorithmSpec(key="failing", label="Failing algorithm", description="", run=failing_run)
    return disambiguate(sentence, target, ["failing"], registry={"failing": spec})


# The MSH-WSD data are not in the repository: SapBERT finds no candidates in these tests.
@patch("algorithms.sapbert.concept_map", new=lambda: {})
class IndexViewTests(SimpleTestCase):
    url = reverse("wsd:index")

    def test_empty_page_shows_the_form_without_errors(self) -> None:
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["form"].is_bound)
        self.assertIsNone(response.context["report"])
        self.assertTemplateUsed(response, "wsd/index.html")
        self.assertTemplateUsed(response, "wsd/base.html")
        self.assertContains(response, "bootstrap.min.css")

    def test_valid_input_shows_the_result(self) -> None:
        params = {"sentence": DRUG_SENTENCE, "target": "drug", "algorithm": "extended_lesk"}
        response = self.client.get(self.url, params)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "drug.n.01")
        self.assertEqual(len(response.context["report"].runs), 1)

    def test_compare_all_runs_every_algorithm(self) -> None:
        params = {"sentence": DRUG_SENTENCE, "target": "drug", "algorithm": "all"}
        response = self.client.get(self.url, params)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["report"].runs), len(ALGORITHMS))
        self.assertIs(response.context["report"].agreement, False)

    def test_invalid_form_shows_errors_and_no_result(self) -> None:
        params = {"sentence": DRUG_SENTENCE, "target": "dog", "algorithm": "lesk"}
        response = self.client.get(self.url, params)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "does not appear in the sentence")
        self.assertIsNone(response.context["report"])

    def test_word_without_synsets(self) -> None:
        params = {"sentence": "He said xyzzy loudly", "target": "xyzzy", "algorithm": "all"}
        response = self.client.get(self.url, params)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No candidate senses.", count=len(ALGORITHMS))

    def test_evidence_is_highlighted(self) -> None:
        params = {"sentence": "I went to the bank to deposit my money", "target": "bank",
                  "algorithm": "extended_lesk_pos"}
        response = self.client.get(self.url, params)
        self.assertContains(response, "<mark>deposit</mark>", html=False)
        self.assertContains(response, "All candidates (10)")

    def test_failing_algorithm_shows_an_error_card(self) -> None:
        params = {"sentence": DRUG_SENTENCE, "target": "drug", "algorithm": "lesk"}
        with (
            patch("wsd.views.disambiguate", disambiguate_with_failing_algorithm),
            self.assertLogs("wsd.services", level="ERROR"),
        ):
            response = self.client.get(self.url, params)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "This algorithm failed.")
        self.assertContains(response, "RuntimeError: boom")

    def test_score_chart_data_and_scripts(self) -> None:
        params = {"sentence": DRUG_SENTENCE, "target": "drug", "algorithm": "lesk"}
        response = self.client.get(self.url, params)
        self.assertContains(response, '<script id="chart-data-lesk" type="application/json">')
        self.assertContains(response, 'data-chart-source="chart-data-lesk"')
        self.assertContains(response, "chart.umd.min.js")
        self.assertContains(response, "wsd/score_charts.js")

    def test_no_chart_scripts_without_results(self) -> None:
        response = self.client.get(self.url)
        self.assertNotContains(response, "chart.umd.min.js")

    def test_disagreement_notice_in_comparison(self) -> None:
        params = {"sentence": DRUG_SENTENCE, "target": "drug", "algorithm": "all"}
        response = self.client.get(self.url, params)
        self.assertContains(response, "The algorithms disagree.")
        self.assertContains(response, 'href="#run-lesk"')
        self.assertContains(response, 'id="run-lesk"')

    def test_no_comparison_notice_for_a_single_algorithm(self) -> None:
        params = {"sentence": DRUG_SENTENCE, "target": "drug", "algorithm": "lesk"}
        response = self.client.get(self.url, params)
        self.assertNotContains(response, "The algorithms disagree.")
        self.assertNotContains(response, "The algorithms agree.")

    def test_runs_without_choice_are_listed(self) -> None:
        params = {"sentence": "He said xyzzy loudly", "target": "xyzzy", "algorithm": "all"}
        response = self.client.get(self.url, params)
        self.assertContains(response, "there is nothing to compare")
        self.assertContains(response, "Not included in the comparison:")

    def test_example_links_and_loading_script(self) -> None:
        response = self.client.get(self.url)
        self.assertContains(response, "?sentence=I+went+to+the+bank+to+deposit+my+money&amp;target=bank&amp;algorithm=all")
        self.assertContains(response, "wsd/form_loading.js")

    def test_readable_pos_and_target_in_evidence(self) -> None:
        params = {"sentence": "I went to the bank to deposit my money", "target": "bank",
                  "algorithm": "extended_lesk_pos"}
        response = self.client.get(self.url, params)
        self.assertContains(response, "POS filter: noun")
        self.assertContains(response, '<span class="fw-bold text-primary" title="Target word">bank</span>')
