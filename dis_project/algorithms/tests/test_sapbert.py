from unittest import TestCase
from unittest.mock import patch

from algorithms.sapbert import SAPBERT_KEY, msh_wsd_word, sapbert, sense_text

DRUG_SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"
VENTRICLE_SENTENCE = "The left ventricle of the heart pumps blood to the body"

# A small stand-in for the MSH-WSD and UMLS data, which are not in the repository.
CONCEPTS = {
    "Nurse": ["C0028661", "C0006147"],
    "Nursing": ["C0028677", "C0006147"],
    "Ventricles": ["C0018827", "C0007799"],
}
UMLS = {
    "C0018827": {"names": ["heart ventricle"],
                 "definitions": ["a chamber of the heart that pumps blood into the arteries"]},
    "C0007799": {"names": ["cerebral ventricles"],
                 "definitions": ["cavities in the brain filled with cerebrospinal fluid"]},
}


@patch("algorithms.sapbert.concept_map", new=lambda: CONCEPTS)
@patch("algorithms.sapbert.umls_info", new=lambda: UMLS)
class SapBertTests(TestCase):
    def test_matches_msh_wsd_word_in_any_form(self) -> None:
        self.assertEqual(msh_wsd_word("ventricle"), "Ventricles")
        self.assertEqual(msh_wsd_word("VENTRICLES"), "Ventricles")
        self.assertIsNone(msh_wsd_word("drug"))

    def test_same_word_comes_before_other_forms(self) -> None:
        # "nursing" is also a form of "nurse"
        self.assertEqual(msh_wsd_word("nursing"), "Nursing")

    def test_sense_text_has_names_and_english_definition(self) -> None:
        self.assertEqual(sense_text("C0018827"),
                         "heart ventricle. a chamber of the heart that pumps blood into the arteries")

    def test_word_not_in_msh_wsd_has_no_candidates(self) -> None:
        with patch("algorithms.sapbert.load_model", side_effect=AssertionError("model loaded")):
            result = sapbert(DRUG_SENTENCE, "drug")
        self.assertEqual(result.ranking, ())
        self.assertIsNone(result.best)

    def test_chooses_the_closest_umls_sense(self) -> None:
        # Downloads the SapBERT model on the first run.
        result = sapbert(VENTRICLE_SENTENCE, "ventricle")
        self.assertEqual(result.algorithm, SAPBERT_KEY)
        self.assertEqual([c.sense.id for c in result.ranking], ["C0018827", "C0007799"])
        self.assertEqual(result.best.details["MSH-WSD word"], "Ventricles")
        self.assertFalse(result.is_fallback)

    def test_target_missing_from_sentence_raises(self) -> None:
        with self.assertRaises(ValueError):
            sapbert("The cat sat on the mat", "dog")
