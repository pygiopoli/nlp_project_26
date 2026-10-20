"""SapBERT embeddings over the UMLS senses of MSH-WSD, ported from scripts/bert_extention/bert_wsd.ipynb.

Only the ambiguous words of MSH-WSD have candidates: their senses are the UMLS
concepts (CUIs) listed in data/MSHCorpus/benchmark_mesh.txt, each described by
its shortest UMLS names and first English definition (data/umls_cui_info.json,
created by `scripts/full_dataset/run_full_dataset.py umls`). The chosen sense
is the one whose SapBERT [CLS] vector is closest (cosine similarity) to the
vector of the WINDOW tokens around the target.

The data are not in the repository (UMLS license, see README): without them
the algorithm fails with FileNotFoundError. The model is downloaded from
Hugging Face on first use. The bodies of embed, is_english and sense_text are
unchanged from the notebook.
"""
import json
import re
from functools import cache
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from algorithms.base import ScoredSense, Sense, WSDResult
from algorithms.nltk_utils import find_target, lemma_forms, rank, stop_words, tokenize_and_tag

SAPBERT_KEY = "sapbert"
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Context tokens on each side of the target (as in the notebook).
WINDOW = 10

# The data/ folder at the repository root (this file is dis_project/algorithms/sapbert.py).
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@cache
def concept_map() -> dict[str, list[str]]:
    """MSH-WSD word -> its candidate CUIs, from benchmark_mesh.txt."""
    concepts = {}
    with open(DATA_DIR / "MSHCorpus" / "benchmark_mesh.txt", encoding="iso-8859-1") as f:
        for line in f:
            word, *cuis = line.strip().split("\t")
            concepts[word] = cuis
    return concepts


@cache
def umls_info() -> dict[str, dict[str, list[str]]]:
    """CUI -> its UMLS names and definitions."""
    return json.loads((DATA_DIR / "umls_cui_info.json").read_text(encoding="utf-8"))


@cache
def load_model() -> tuple[AutoTokenizer, AutoModel]:
    """SapBERT tokenizer and model; we only use the model, we do not train it."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def embed(texts: list[str], batch_size: int = 32) -> torch.Tensor:
    """One vector per text: the [CLS] vector of SapBERT, normalized to length 1."""
    tokenizer, model = load_model()
    vectors = []
    with torch.no_grad():                      # no training, so no gradients (faster, less memory)
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                              max_length=128, return_tensors="pt")
            cls = model(**batch).last_hidden_state[:, 0]          # vector of the [CLS] token
            vectors.append(torch.nn.functional.normalize(cls, dim=1))
    return torch.cat(vectors)


def is_english(text: str) -> bool:
    """True if at least 20% of the words are common English words (the, of, which...)."""
    words = re.findall(r"\w+", text.lower())
    return bool(words) and sum(w in stop_words() for w in words) / len(words) >= 0.2


def sense_text(cui: str, n_names: int = 5) -> str:
    """Description of a sense: its shortest UMLS names and its first English definition."""
    info = umls_info().get(cui, {"names": [], "definitions": []})
    names = sorted(info["names"], key=len)[:n_names]
    english = [d for d in info["definitions"] if is_english(d)]
    return "; ".join(names) + (". " + english[0] if english else "")


def msh_wsd_word(target: str) -> str | None:
    """The MSH-WSD word that matches the target: the same word ignoring case, or else
    another form of it ("ventricle" -> "Ventricles"). None if there is no such word."""
    words = concept_map()
    exact = [word for word in words if word.lower() == target.lower()]
    if exact:
        return exact[0]
    forms = lemma_forms(target)
    return next((word for word in words if lemma_forms(word) & forms), None)


def sapbert(sentence: str, target: str) -> WSDResult:
    """Disambiguate `target` in `sentence` among the UMLS senses of MSH-WSD.

    The target must appear in the sentence (ValueError otherwise). A word that
    is not in MSH-WSD has no candidates, and the model is not loaded.
    """
    tagged = tokenize_and_tag(sentence)
    match = find_target(tagged, target)
    if match is None:
        raise ValueError(f"{target!r} does not appear in the sentence")
    idx, _tok, _pos = match
    word = msh_wsd_word(target)
    if word is None:
        return WSDResult(algorithm=SAPBERT_KEY, target=target, ranking=())

    cuis = concept_map()[word]
    tokens = [tok for tok, _tag in tagged]
    context = " ".join(tokens[max(0, idx - WINDOW):idx + WINDOW + 1])
    descriptions = [sense_text(cui) for cui in cuis]
    vectors = embed([context] + descriptions)
    similarities = (vectors[1:] @ vectors[0]).tolist()   # cosine similarity of each sense with the context

    details = {"MSH-WSD word": word, "context window": context}
    scored = [
        ScoredSense(Sense(cui, "", description), similarity, details=details)
        for cui, description, similarity in zip(cuis, descriptions, similarities)
    ]
    ranking, is_fallback = rank(scored)
    return WSDResult(algorithm=SAPBERT_KEY, target=target, ranking=ranking, is_fallback=is_fallback)
