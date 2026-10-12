from unittest.mock import patch

from django.test import SimpleTestCase
from django.urls import reverse

from algorithms.base import AlgorithmSpec, WSDResult
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
        self.assertEqual(len(response.context["report"].runs), 4)
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
        self.assertContains(response, "No candidate senses.", count=4)

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
