"""
lesk.py - Lesk classico e avanzato che restituiscono anche punteggi e overlap.

NOTA: questa è un'implementazione di prova, così l'app funziona subito.
Per usare il TUO algoritmo basta far sì che lesk_classico() e lesk_avanzato()
restituiscano un LeskResult (vedi dataclass sotto): il resto dell'app non cambia.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------- risorse NLTK
_RISORSE = {
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
    "corpora/stopwords": "stopwords",
    "tokenizers/punkt_tab": "punkt_tab",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
}


def ensure_nltk_data() -> None:
    """Scarica le risorse NLTK mancanti (solo la prima volta)."""
    for path, nome in _RISORSE.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(nome, quiet=True)


# ------------------------------------------------------------------ strutture
@dataclass
class SenseScore:
    synset: object          # nltk Synset
    score: int              # numero di parole in overlap
    overlap: list[str]      # parole in comune tra contesto e signature


@dataclass
class LeskResult:
    algorithm: str
    word: str
    pos: str | None         # POS WordNet usato per filtrare (solo avanzato)
    ranking: list[SenseScore]   # ordinato per score decrescente

    @property
    def best(self) -> SenseScore | None:
        return self.ranking[0] if self.ranking else None


# ------------------------------------------------------------------- utilità
_lemmatizer = WordNetLemmatizer()


def _stop() -> set[str]:
    return set(stopwords.words("english"))


def _penn_to_wn(tag: str) -> str | None:
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("R"):
        return wn.ADV
    return None


def analizza_frase(frase: str) -> list[tuple[str, str, str | None]]:
    """Restituisce [(token, tag Penn, POS WordNet)] per ogni token."""
    tokens = nltk.word_tokenize(frase)
    return [(t, tag, _penn_to_wn(tag)) for t, tag in nltk.pos_tag(tokens)]


def forme_possibili(parola: str) -> set[str]:
    """Forma minuscola + lemmi (n/v/a): serve a evidenziare gli overlap."""
    p = parola.lower()
    return {p, _lemmatizer.lemmatize(p, "n"), _lemmatizer.lemmatize(p, "v"),
            _lemmatizer.lemmatize(p, "a")}


def _tokens_semplici(testo: str) -> set[str]:
    """Tokenizzazione base: minuscolo, solo lettere, senza stopword."""
    stop = _stop()
    return {t for t in re.findall(r"[a-z]+", testo.lower()) if t not in stop}


def _tokens_lemmatizzati(testo: str) -> set[str]:
    """Come sopra, ma con POS tagging + lemmatizzazione."""
    stop = _stop()
    out = set()
    for tok, _tag, pos in analizza_frase(testo):
        t = tok.lower()
        if not t.isalpha() or t in stop:
            continue
        out.add(_lemmatizer.lemmatize(t, pos) if pos else t)
    return out


def _classifica(contesto: set[str], synsets, signature_fn) -> list[SenseScore]:
    ranking = []
    for s in synsets:
        comuni = sorted(contesto & signature_fn(s))
        ranking.append(SenseScore(s, len(comuni), comuni))
    # sort stabile: a parità di score resta l'ordine di frequenza di WordNet
    ranking.sort(key=lambda r: -r.score)
    return ranking


# ------------------------------------------------------------- Lesk classico
def lesk_classico(frase: str, parola: str) -> LeskResult:
    """Lesk semplificato: overlap tra contesto e sola definizione, tutti i POS."""
    contesto = _tokens_semplici(frase) - {parola.lower()}
    synsets = wn.synsets(parola)
    ranking = _classifica(contesto, synsets,
                          lambda s: _tokens_semplici(s.definition()))
    return LeskResult("Lesk classico", parola, None, ranking)


# ------------------------------------------------------------- Lesk avanzato
def _signature_estesa(s) -> set[str]:
    testo = [s.definition(), *s.examples()]
    for h in s.hypernyms():
        testo.append(h.definition())
        testo.extend(l.replace("_", " ") for l in h.lemma_names())
    return _tokens_lemmatizzati(" ".join(testo))


def lesk_avanzato(frase: str, parola: str) -> LeskResult:
    """Lesk con POS tagging, lemmatizzazione e glossa estesa (def + esempi + iperonimi)."""
    analisi = analizza_frase(frase)
    pos_target = None
    for tok, _tag, pos in analisi:
        if forme_possibili(tok) & forme_possibili(parola):
            pos_target = pos
            break

    contesto = _tokens_lemmatizzati(frase) - forme_possibili(parola)
    synsets = wn.synsets(parola, pos=pos_target) if pos_target else wn.synsets(parola)
    if not synsets:                       # fallback se il filtro POS svuota tutto
        synsets = wn.synsets(parola)
    ranking = _classifica(contesto, synsets, _signature_estesa)
    return LeskResult("Lesk avanzato", parola, pos_target, ranking)
