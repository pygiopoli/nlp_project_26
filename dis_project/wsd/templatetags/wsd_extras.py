"""Template filters for the WSD pages."""
from django import template

register = template.Library()

# WordNet POS codes and their names ("s" is WordNet's "adjective satellite").
WORDNET_POS_NAMES = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",
    "r": "adverb",
}


@register.filter
def pos_name(pos: str | None) -> str:
    """Readable name of a WordNet POS code: "n" -> "noun"; unknown codes are returned as they are."""
    if not pos:
        return ""
    return WORDNET_POS_NAMES.get(pos, pos)
