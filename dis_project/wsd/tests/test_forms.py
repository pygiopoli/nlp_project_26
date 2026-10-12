from django.test import SimpleTestCase

from algorithms.registry import ALGORITHMS
from wsd.forms import COMPARE_ALL, MAX_SENTENCE_LENGTH, DisambiguationForm

DRUG_SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"


def make_form(**overrides: str) -> DisambiguationForm:
    data = {"sentence": DRUG_SENTENCE, "target": "drug", "algorithm": "lesk", **overrides}
    return DisambiguationForm(data)


class DisambiguationFormTests(SimpleTestCase):
    def test_valid_input(self) -> None:
        form = make_form()
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.selected_algorithm_keys(), ["lesk"])

    def test_choices_come_from_the_registry(self) -> None:
        keys = [key for key, _label in DisambiguationForm().fields["algorithm"].choices]
        self.assertEqual(keys, [*ALGORITHMS, COMPARE_ALL])

    def test_compare_all_selects_every_algorithm(self) -> None:
        form = make_form(algorithm=COMPARE_ALL)
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.selected_algorithm_keys(), list(ALGORITHMS))

    def test_compare_all_key_is_not_an_algorithm(self) -> None:
        self.assertNotIn(COMPARE_ALL, ALGORITHMS)

    def test_unknown_algorithm_is_rejected(self) -> None:
        self.assertIn("algorithm", make_form(algorithm="nope").errors)

    def test_inflected_target_is_accepted(self) -> None:
        self.assertTrue(make_form(target="Drugs").is_valid())

    def test_target_must_be_a_single_word(self) -> None:
        form = make_form(target="important drugs")
        self.assertEqual(form.errors["target"], ["Enter a single word."])

    def test_target_must_appear_in_the_sentence(self) -> None:
        form = make_form(target="dog")
        self.assertEqual(form.errors["target"], ["“dog” does not appear in the sentence."])

    def test_sentence_too_long(self) -> None:
        self.assertIn("sentence", make_form(sentence="a" * (MAX_SENTENCE_LENGTH + 1)).errors)

    def test_blank_fields_are_required(self) -> None:
        form = make_form(sentence="   ", target="")
        self.assertIn("sentence", form.errors)
        self.assertIn("target", form.errors)
