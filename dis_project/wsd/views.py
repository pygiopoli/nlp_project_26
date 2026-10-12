from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from wsd.forms import DisambiguationForm
from wsd.services import disambiguate


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
    return render(request, "wsd/index.html", {"form": form, "report": report})
