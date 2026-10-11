"""
Phonetic extension of (Extended) Lesk, following section 4.2 of
"Ultimate Guide to Understand and Implement NLP (with codes in Python)"
(Analytics Vidhya):
    4.2-A  Levenshtein distance      -> distance between two strings
    4.2-B  Phonetic matching/Soundex -> fuzzy.Soundex(4) gives a key that
                                        identifies phonetically similar words

Pipeline
  1. fuzzy.Soundex(4) turns every word into a 4-character phonetic key
  2. Levenshtein distance between two keys = phonetic distance of the two words
  3. a distance matrix is computed for every pair of words in the sentence
  4. Extended Lesk is re-run, counting phonetically similar words as overlaps
"""
import fuzzy
import nltk
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab",
            "averaged_perceptron_tagger_eng", "stopwords"):
    nltk.download(pkg, quiet=True)

SENTENCE = "I have been prescribed two important drugs today during my visit to clinic"
TARGET = "drug"
OUT = "/mnt/user-data/outputs"

lemmatizer = WordNetLemmatizer()
STOP = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# 1) Phonetic key: fuzzy.Soundex(4), exactly as in the article
# ---------------------------------------------------------------------------
def _replica_soundex(word, size=4):
    """Pure-Python replica of fuzzy.Soundex's algorithm (used only if the
    compiled fuzzy.Soundex is broken, see below)."""
    MAP = "01230120022455012623010202"
    out = ""
    for ch in word.upper():
        if "A" <= ch <= "Z":
            if not out:
                out = ch
            else:
                code = MAP[ord(ch) - 65]
                if code != "0" and (len(out) == 1 or out[-1] != code):
                    out += code
        if len(out) == size:
            break
    return out.ljust(size, "0")


def _fuzzy_soundex_works():
    # Fuzzy 1.2.2 has a use-after-free bug in Soundex.__call__ (the C buffer is
    # free()d before being returned), so on some builds it returns garbage.
    # Article's own example: soundex('ankit') == soundex('aunkit') == 'A523'.
    try:
        s = fuzzy.Soundex(4)
        return s("ankit") == "A523" and s("aunkit") == "A523" and s("drugs") == "D620"
    except Exception:
        return False


FUZZY_OK = _fuzzy_soundex_works()
_soundex = fuzzy.Soundex(4) if FUZZY_OK else None
_cache = {}


def phonetic_key(word):
    w = word.lower()
    if w not in _cache:
        _cache[w] = _soundex(w) if FUZZY_OK else _replica_soundex(w)
    return _cache[w]


# ---------------------------------------------------------------------------
# 2) Phonetic distance = Levenshtein distance between the phonetic keys
# ---------------------------------------------------------------------------
def phonetic_distance(w1, w2):
    return nltk.edit_distance(phonetic_key(w1), phonetic_key(w2))


def distance_matrix(words):
    return pd.DataFrame([[phonetic_distance(a, b) for b in words] for a in words],
                        index=words, columns=words)


# ---------------------------------------------------------------------------
# 3) Extended Lesk + phonetic matching
# ---------------------------------------------------------------------------
def penn_to_wn(tag):
    return {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}.get(tag[0])


def normalize(text):
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text.lower())
            if w.isalpha() and w not in STOP]


def related_synsets(s, depth=2):
    rel = list(s.hypernyms())
    level = s.hyponyms()
    for _ in range(depth):
        rel += level
        level = [h for x in level for h in x.hyponyms()]
    rel += s.part_meronyms() + s.member_meronyms() + s.substance_meronyms()
    rel += s.part_holonyms() + s.member_holonyms() + s.substance_holonyms()
    rel += s.similar_tos() + s.also_sees() + s.attributes()
    rel += s.entailments() + s.verb_groups()
    return rel


def extended_gloss(s):
    words = normalize(s.definition())
    for ex in s.examples():
        words += normalize(ex)
    for r in related_synsets(s):
        words += [l.lower() for n in r.lemma_names() for l in n.split("_")]
        words += normalize(r.definition())
        for ex in r.examples():
            words += normalize(ex)
    return set(words)


def context_lemmas(tagged, idx):
    return {lemmatizer.lemmatize(t.lower(), penn_to_wn(tag) or wn.NOUN)
            for i, (t, tag) in enumerate(tagged)
            if i != idx and t.isalpha() and t.lower() not in STOP}


def phonetic_overlap(context, gloss, max_dist=0):
    """Exact overlaps + phonetic-only overlaps (Levenshtein distance between the
    Soundex keys <= max_dist). Returns (exact_set, {context_word: [gloss words]})."""
    exact = context & gloss
    phon = {}
    for c in context - exact:
        kc = phonetic_key(c)
        hits = sorted(g for g in gloss - context
                      if nltk.edit_distance(kc, phonetic_key(g)) <= max_dist)
        if hits:
            phon[c] = hits
    return exact, phon


def phonetic_lesk(tagged, idx, pos=None, max_dist=0, w_phon=0.5):
    lemma = lemmatizer.lemmatize(tagged[idx][0].lower(), pos or wn.NOUN)
    ctx = context_lemmas(tagged, idx)
    res = {}
    for s in wn.synsets(lemma, pos=pos):
        exact, phon = phonetic_overlap(ctx, extended_gloss(s), max_dist)
        res[s] = (len(exact) + w_phon * len(phon), exact, phon)
    return res


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tokens = word_tokenize(SENTENCE)
    tagged = pos_tag(tokens)
    idx = next(i for i, (t, tg) in enumerate(tagged)
               if penn_to_wn(tg) and lemmatizer.lemmatize(t.lower(), penn_to_wn(tg)) == TARGET)
    wnpos = penn_to_wn(tagged[idx][1])

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    print("fuzzy.Soundex usable as-is:", FUZZY_OK,
          "" if FUZZY_OK else "(using an exact pure-Python replica instead)")
    if FUZZY_OK:
        print("fuzzy == replica on all tokens:",
              all(_soundex(t.lower()) == _replica_soundex(t) for t in tokens))
    print("Article check: soundex('ankit') =", phonetic_key("ankit"),
          "| soundex('aunkit') =", phonetic_key("aunkit"), "\n")

    # --- phonetic keys -------------------------------------------------------
    keys = pd.DataFrame({"word": tokens, "Soundex": [phonetic_key(w) for w in tokens]})
    print(keys.set_index("word").T.to_string(), "\n")

    # --- distance matrix -----------------------------------------------------
    M = distance_matrix(tokens)
    print("Phonetic distance matrix (Levenshtein distance between Soundex keys)\n")
    print(M, "\n")
    M.to_csv(f"{OUT}/phonetic_distance_matrix.csv")

    pairs = sorted((int(M.iloc[i, j]), tokens[i], tokens[j])
                   for i in range(len(tokens)) for j in range(i + 1, len(tokens)))
    print("Pairs with distance 0 (same Soundex key):",
          [(a, b) for d, a, b in pairs if d == 0])
    print("Pairs with distance 1:", [(a, b) for d, a, b in pairs if d == 1], "\n")

    # --- heat-map ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M.values, cmap="viridis_r", vmin=0, vmax=4)
    ax.set_xticks(range(len(tokens)), tokens, rotation=60, ha="right")
    ax.set_yticks(range(len(tokens)), tokens)
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, M.iloc[i, j], ha="center", va="center", fontsize=8,
                    color="white" if M.iloc[i, j] >= 3 else "black")
    ax.set_title("Phonetic distance (Levenshtein on Soundex keys)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/phonetic_distance_matrix.png", dpi=150)

    # --- Phonetic Extended Lesk ---------------------------------------------
    print(f"Target: '{tagged[idx][0]}' -> WordNet POS '{wnpos}'")
    print("Context lemmas:", sorted(context_lemmas(tagged, idx)), "\n")
    for max_dist in (0, 1):
        for label, pos in (("with POS filter", wnpos), ("no POS filter", None)):
            print(f"Phonetic Extended Lesk | Soundex distance <= {max_dist} | {label}")
            res = phonetic_lesk(tagged, idx, pos=pos, max_dist=max_dist)
            for s, (score, exact, phon) in res.items():
                shown = {c: (g[:4] + ["..."] if len(g) > 4 else g) for c, g in phon.items()}
                print(f"  {s.name():<11} score={score:<4} exact={sorted(exact)} phonetic={shown}")
            best = max(res, key=lambda s: res[s][0])
            print("  ->", best, "-", best.definition(), "\n")
