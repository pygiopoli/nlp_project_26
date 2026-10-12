from django import forms

from algorithms.nltk_utils import find_target, tokenize_and_tag
from algorithms.registry import ALGORITHMS

MAX_SENTENCE_LENGTH = 500
MAX_TARGET_LENGTH = 50

# Reserved choice that runs every registered algorithm (comparison mode).
COMPARE_ALL = "all"


def algorithm_choices() -> list[tuple[str, str]]:
    """One choice per registered algorithm, plus the comparison of all of them."""
    choices = [(key, spec.label) for key, spec in ALGORITHMS.items()]
    return [*choices, (COMPARE_ALL, "Compare all algorithms")]


class DisambiguationForm(forms.Form):
    # The "class" attributes are Bootstrap CSS classes.
    sentence = forms.CharField(
        max_length=MAX_SENTENCE_LENGTH,
        widget=forms.Textarea(attrs={
            "rows": 2,
            "class": "form-control",
            "placeholder": "e.g. I went to the bank to deposit my money",
        }),
    )
    target = forms.CharField(
        max_length=MAX_TARGET_LENGTH,
        label="Target word",
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "e.g. bank"}),
    )
    algorithm = forms.ChoiceField(
        choices=algorithm_choices,
        initial=COMPARE_ALL,
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    def clean_target(self) -> str:
        target = self.cleaned_data["target"]
        if len(target.split()) > 1:
            raise forms.ValidationError("Enter a single word.")
        return target

    def clean(self) -> dict:
        cleaned_data = super().clean()
        sentence = cleaned_data.get("sentence")
        target = cleaned_data.get("target")
        # Only check when both fields are individually valid.
        if sentence and target and find_target(tokenize_and_tag(sentence), target) is None:
            self.add_error("target", f"“{target}” does not appear in the sentence.")
        return cleaned_data

    def selected_algorithm_keys(self) -> list[str]:
        """The keys of the algorithms to run (call only after is_valid())."""
        choice = self.cleaned_data["algorithm"]
        return list(ALGORITHMS) if choice == COMPARE_ALL else [choice]
