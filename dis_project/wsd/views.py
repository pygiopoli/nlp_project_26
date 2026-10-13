from urllib.parse import urlencode

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.urls import reverse

from wsd.forms import COMPARE_ALL, DisambiguationForm
from wsd.services import disambiguate

# (sentence, target) pairs offered as one-click examples under the form.
EXAMPLES: list[tuple[str, str]] = [
    ("I have been prescribed two important drugs today during my visit to clinic", "drug"),
    ("I went to the bank to deposit my money", "bank"),
    ("The patient has a cold", "cold"),
    ("The left ventricle of the heart pumps blood to the body", "ventricle"),
]


def example_links() -> list[dict[str, str]]:
    """A link for each example, comparing all the algorithms."""
    base_url = reverse("wsd:index")
    return [
        {
            "target": target,
            "sentence": sentence,
            "url": f"{base_url}?{urlencode({'sentence': sentence, 'target': target, 'algorithm': COMPARE_ALL})}",
        }
        for sentence, target in EXAMPLES
    ]


def index(request: HttpRequest) -> HttpResponse:
    """Show the form and, when it is submitted and valid, the results."""
    # An empty query string gives an unbound form, so no errors are shown.
    form = DisambiguationForm(request.GET or None)
    report = None
    if form.is_valid():
        report = disambiguate(
            form.cleaned_data["sentence"],
            form.cleaned_data["target"],
            form.selected_algorithm_keys(),
        )
    context = {"form": form, "report": report, "examples": example_links()}
    return render(request, "wsd/index.html", context)
