"""Download the NLTK data needed by the algorithms.

Run once after installing the requirements:

    python -m algorithms.nltk_data

The data go to NLTK's default user folder (e.g. %APPDATA%\\nltk_data on Windows).
"""
import nltk

REQUIRED_NLTK_DATA: tuple[str, ...] = (
    "wordnet",
    "stopwords",
    "punkt_tab",                        # used by nltk.word_tokenize
    "averaged_perceptron_tagger_eng",   # used by nltk.pos_tag
)


def download_nltk_data() -> None:
    """Download every required package; up-to-date packages are skipped."""
    for package in REQUIRED_NLTK_DATA:
        nltk.download(package, quiet=True)


if __name__ == "__main__":
    download_nltk_data()
    print("NLTK data ready:", ", ".join(REQUIRED_NLTK_DATA))
