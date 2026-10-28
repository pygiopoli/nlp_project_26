"""
Task 4 and Task 6 on the FULL MSH-WSD dataset (all words), with an automatic CUI -> WordNet mapping.

Run one step at a time from this folder:
    python run_full_dataset.py info       # number of words and CUIs (quick check)
    python run_full_dataset.py umls       # download UMLS names and definitions of all CUIs
    python run_full_dataset.py mapping    # automatic mapping + validation on our 39 manual words
    python run_full_dataset.py task4      # Task 4 methods on all words
    python run_full_dataset.py task6      # Task 6 (Wu-Palmer + phonetic) on all words
    python run_full_dataset.py results    # result tables of Task 4 and Task 6

Every step saves its results in results/, so it can be interrupted and restarted.
"""
import sys
import json
import getpass
from pathlib import Path
import pandas as pd 
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import time
from nltk.wsd import lesk
from pywsd.lesk import original_lesk, simple_lesk, adapted_lesk, cosine_lesk
from pywsd.similarity import max_similarity
from pywsd.baseline import max_lemma_count
from pywsd._wordnet import wn as pwn
from pywsd.similarity import sim
from pywsd.utils import lemmatize
from pywsd.tokenize import word_tokenize

sys.path.append("../p5_6_10_phonetic")          # utils.py of Task 5 (Soundex and Levenshtein)
import utils as u

import requests

sys.path.append("..")                       
from wsd_utils import (CONCEPT_MAP, ALL_ACRONYMS, ALL_TERMS, DATA_DIR, wordnet_senses, MAPPING,
                       to_nltk, pywsd_form, context_window, sample_instances, accuracy, mean)
RESULTS = Path("results")                   
RESULTS.mkdir(exist_ok=True)

ALL_CUIS = sorted({cui for senses in CONCEPT_MAP.values() for cui in senses.values()})
UMLS_FILE = DATA_DIR.parent / "umls_cui_info.json"   

def step_info():
    """Quick check: how many words and CUIs we have to process."""
    all_cuis = {cui for senses in CONCEPT_MAP.values() for cui in senses.values()}
    print("Acronyms:", len(ALL_ACRONYMS), "  Terms:", len(ALL_TERMS))
    print("CUIs    :", len(all_cuis))

UTS = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI"

def umls_info(cui, api_key):
    """English names (preferred name + synonyms) and definitions of a CUI, lowercase."""
    info = {"names": [], "definitions": []}
    r = requests.get(f"{UTS}/{cui}/atoms", params={"apiKey": api_key, "language": "ENG", "pageSize": 200}, timeout=30)
    if r.ok:                                # some 2009 CUIs no longer exist in the current UMLS
        info["names"] = sorted({atom["name"].lower() for atom in r.json()["result"]})
    r = requests.get(f"{UTS}/{cui}/definitions", params={"apiKey": api_key, "pageSize": 50}, timeout=30)
    if r.ok:                                # many CUIs have no definition
        info["definitions"] = [d["value"].lower() for d in r.json()["result"]]
    return info


def step_umls():
    """Download the UMLS names and definitions of all the CUIs (saved every 20 CUIs)."""
    umls = json.loads(UMLS_FILE.read_text(encoding="utf-8")) if UMLS_FILE.exists() else {}
    todo = [c for c in ALL_CUIS if c not in umls]
    print(len(ALL_CUIS), "CUIs,", len(todo), "still to download")
    if todo:
        api_key = getpass.getpass("UMLS API key: ")
    for i, cui in enumerate(todo, 1):
        umls[cui] = umls_info(cui, api_key)
        if i % 20 == 0 or i == len(todo):
            UMLS_FILE.write_text(json.dumps(umls, indent=1), encoding="utf-8")
            print(f"   {i}/{len(todo)}")
    print(sum(1 for c in ALL_CUIS if not umls[c]["names"]), "CUIs not found,",
          sum(1 for c in ALL_CUIS if not umls[c]["definitions"]), "without definition")
    print("Example C0015422:", umls["C0015422"])


STOP = set(stopwords.words("english"))
MAPPING_FILE = RESULTS / "auto_mapping.json"
MIN_OVERLAP = 4

def is_english(text):
    """True if at least 20% of the words are common English words (the, of, which...)."""
    words = re.findall(r"\w+", text.lower())
    return bool(words) and sum(w in STOP for w in words) / len(words) >= 0.2


def normalize(text):
    """Lowercase base forms: 'Eyelashes' -> 'eyelash', 'adenosine_diphosphate' -> 'adenosine diphosphate'."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(wn.morphy(t, "n") or wn.morphy(t, "v") or t for t in tokens)


def content_words(text, excluded):
    """Meaningful words of a text: normalized, without stopwords, short words and the target word."""
    return {w for w in normalize(text).split() if w not in STOP and len(w) > 2 and w not in excluded}

def auto_mapping_word(word, umls, min_overlap=MIN_OVERLAP):
    """Automatic CUI -> synsets mapping of one word (same format as MAPPING).

    Rule 1 (names): a WordNet sense goes to a CUI if one of its names (not the word itself)
                    is also a UMLS name of the CUI, e.g. eyelash.n.01 -> 'eyelash'.
    Rule 2 (words): otherwise, the CUI whose UMLS names and English definitions share most
                    words with the WordNet definition (at least MIN_OVERLAP).
    The winner must be unique, so each sense goes to at most one CUI.
    """
    cuis = list(CONCEPT_MAP[word].values())
    target = set(normalize(word).split())              # the word itself gives no information
    names = {c: {normalize(n) for n in umls[c]["names"]} for c in cuis}
    words = {}
    for c in cuis:
        english_defs = [d for d in umls[c]["definitions"] if is_english(d)]
        words[c] = content_words(" ".join(umls[c]["names"] + english_defs), target)

    mapping = {c: [] for c in cuis}
    for s in wordnet_senses(word):
        lemmas = {normalize(l) for l in s.lemma_names()} - {normalize(word)}
        matches = [c for c in cuis if lemmas & names[c]]                        # rule 1
        if len(matches) != 1:
            bag = content_words(s.definition() + " " + " ".join(s.lemma_names()), target)
            for h in s.hypernyms():
                bag |= content_words(" ".join(h.lemma_names()), target)
            scores = {c: len(bag & words[c]) for c in cuis}                      # rule 2
            best = max(scores.values())
            matches = [c for c in cuis if scores[c] == best] if best >= min_overlap else []
        if len(matches) == 1:
            mapping[matches[0]].append(s.name())
    return mapping

def step_mapping(min_overlap=MIN_OVERLAP):
    """Automatic mapping of all the words, saved in results/auto_mapping.json."""
    umls = json.loads(UMLS_FILE.read_text(encoding="utf-8"))
    auto = {w: auto_mapping_word(w, umls, min_overlap) for w in ALL_ACRONYMS + ALL_TERMS}
    print("min_overlap =", min_overlap)
    MAPPING_FILE.write_text(json.dumps(auto, indent=1), encoding="utf-8")
    n_covered = sum(bool(synsets) for m in auto.values() for synsets in m.values())
    n_words = sum(any(m.values()) for m in auto.values())
    print(f"{n_covered} of {len(ALL_CUIS)} CUIs have at least one WordNet sense")
    print(f"{n_words} of {len(auto)} words have at least one covered CUI")
    print("Example ALS:", auto["ALS"])
    
    validate(auto)

def validate(auto):
    """Compare the automatic mapping with our manual MAPPING (Task 2) on the 39 manual words."""
    manual_pairs = {(w, c, s) for w in MAPPING for c, synsets in MAPPING[w].items() for s in synsets}
    auto_pairs = {(w, c, s) for w in MAPPING for c, synsets in auto[w].items() for s in synsets}
    correct = auto_pairs & manual_pairs
    same_coverage = [bool(auto[w][c]) == bool(MAPPING[w][c]) for w in MAPPING for c in MAPPING[w]]

    print(f"\nValidation on the {len(MAPPING)} manual words:")
    print(f"   precision {len(correct) / len(auto_pairs):.2f}   (automatic pairs that are also in our mapping)")
    print(f"   recall    {len(correct) / len(manual_pairs):.2f}   (our pairs found by the automatic mapping)")
    p, r = len(correct) / len(auto_pairs), len(correct) / len(manual_pairs)
    print(f"   F1        {2 * p * r / (p + r):.2f}   (balance between precision and recall)")
    print(f"   coverage agreement {sum(same_coverage) / len(same_coverage):.2f}   (same decision 'in WordNet or not')")
    print("\n   Wrong (only automatic):")
    for pair in sorted(auto_pairs - manual_pairs):
        print("     ", pair)
    print("\n   Missing (only manual):")
    for pair in sorted(manual_pairs - auto_pairs):
        print("     ", pair)

# ---------------------------------------------------------------- Task 4 on all the words --------------------------------------------------------------------------- # 

TASK4_FILE = RESULTS / "task4_predictions.json"


def resnik_fixed(context, word):
    """Resnik with the ranking sorted correctly (highest score = most similar)."""
    ranking = max_similarity(context, word, option="res", best=False)
    if not ranking:
        return None
    return max(ranking, key=lambda x: x[0])[1]


METHODS = {
    "Original Lesk":    lambda ctx, w: original_lesk(ctx, w),
    "Simple Lesk":      lambda ctx, w: simple_lesk(ctx, w),
    "Adapted Lesk":     lambda ctx, w: adapted_lesk(ctx, w),
    "Cosine Lesk":      lambda ctx, w: cosine_lesk(ctx, w),
    "Path":             lambda ctx, w: max_similarity(ctx, w, option="path"),
    "Wu-Palmer":        lambda ctx, w: max_similarity(ctx, w, option="wup"),
    "Leacock-Chodorow": lambda ctx, w: max_similarity(ctx, w, option="lch"),
    "Resnik (pywsd)":   lambda ctx, w: max_similarity(ctx, w, option="res"),
    "Resnik (fixed)":   resnik_fixed,
    "Jiang-Conrath":    lambda ctx, w: max_similarity(ctx, w, option="jcn"),
    "Lin":              lambda ctx, w: max_similarity(ctx, w, option="lin"),
    "Max lemma count":  lambda ctx, w: max_lemma_count(w),
}
ALL_METHODS = ["NLTK Lesk"] + list(METHODS)


def predict(method, context, word, form):
    """Answer of one method on one context, as an NLTK synset name (None if no answer)."""
    try:
        if method == "NLTK Lesk":
            senses = wordnet_senses(word)
            if not senses:
                return None
            answer = lesk(context.lower().split(), word.lower(), synsets=senses)
            return answer.name() if answer else None
        return to_nltk(METHODS[method](context, form))
    except Exception:
        return None

def step_task4():
    """Run all the Task 4 methods on the covered words (saved after every word)."""
    mapping = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    predictions = json.loads(TASK4_FILE.read_text(encoding="utf-8")) if TASK4_FILE.exists() else {}

    words = [w for w in ALL_ACRONYMS + ALL_TERMS if any(mapping[w].values())]   # covered words only
    todo = [w for w in words if w not in predictions]
    print(len(words), "covered words,", len(todo), "still to do")

    start = time.time()
    for i, word in enumerate(todo, 1):
        t = time.time()
        form = pywsd_form(word, mapping)          # spelling for pywsd, computed once per word
        rows = []
        for inst in sample_instances(word):
            ctx = context_window(inst["tagged"])
            answers = {m: predict(m, ctx, word, form) for m in ALL_METHODS}
            rows.append({"cui": inst["cui"], "answers": answers})
        predictions[word] = rows
        TASK4_FILE.write_text(json.dumps(predictions), encoding="utf-8")     # save after every word
        print(f"[{i:3}/{len(todo)}] {word:20} {len(rows):3} abstracts {time.time() - t:6.1f} s"
              f"   (total {(time.time() - start) / 60:.0f} min)")
    print("Done.")

# ---------------------------------------------------------------- Task 6 on all the words

TASK6_FILE = RESULTS / "task6_predictions.json"
ALPHAS = [1, 0.8, 0.5]


def phonetic_similarity(w1, w2):
    """Phonetic similarity between two words, from 0 to 1 (Soundex codes have 4 characters)."""
    return 1 - u.levenshtein(u.soundex(w1), u.soundex(w2)) / 4


def sense_phonetic(s1, s2):
    """Phonetic similarity between two senses: the best value among all the pairs of their lemmas."""
    return max(phonetic_similarity(a, b) for a in s1.lemmas() for b in s2.lemmas())


def wsd_alphas(context, word):
    """Best sense for each alpha. Wu-Palmer and phonetic values are computed only once and reused."""
    ctx = [lemmatize(w) for w in word_tokenize(context)]
    target = lemmatize(word)
    ctx = [w for w in ctx if w.lower() != target.lower()]
    totals = {a: [] for a in ALPHAS}
    for i in pwn.synsets(target):
        pairs = [[(sim(i, k, "wup") or 0.0, sense_phonetic(i, k)) for k in pwn.synsets(j)] for j in ctx]
        for a in ALPHAS:
            total = sum(max([a * w + (1 - a) * p for w, p in pairs_j], default=0.0) for pairs_j in pairs)
            totals[a].append((total, i))
    return {str(a): to_nltk(max(t, key=lambda x: x[0])[1]) if t else None for a, t in totals.items()}


def step_task6():
    """Run Wu-Palmer + phonetic (all alphas) on the covered words (saved after every word)."""
    mapping = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    predictions = json.loads(TASK6_FILE.read_text(encoding="utf-8")) if TASK6_FILE.exists() else {}

    words = [w for w in ALL_ACRONYMS + ALL_TERMS if any(mapping[w].values())]   # covered words only
    todo = [w for w in words if w not in predictions]
    print(len(words), "covered words,", len(todo), "still to do")

    start = time.time()
    for i, word in enumerate(todo, 1):
        t = time.time()
        form = pywsd_form(word, mapping)
        rows = []
        for inst in sample_instances(word):
            answers = wsd_alphas(context_window(inst["tagged"]), form)
            rows.append({"cui": inst["cui"], "answers": answers})
        predictions[word] = rows
        TASK6_FILE.write_text(json.dumps(predictions), encoding="utf-8")
        print(f"[{i:3}/{len(todo)}] {word:20} {len(rows):3} abstracts {time.time() - t:6.1f} s"
              f"   (total {(time.time() - start) / 60:.0f} min)")
    print("Done.")

# ---------------------------------------------------------------- Results -------------------------------------------------------------------------------------------- #

def group_scores(predictions, mapping, words, method):
    """Accuracy and accuracy on covered senses of one method, averaged over a group of words.
    Words with no sense in WordNet count as 0 in the accuracy (every answer is wrong)."""
    acc, cov = [], []
    for w in words:
        if w in predictions:
            acc.append(accuracy(predictions[w], w, method, mapping=mapping))
            cov.append(accuracy(predictions[w], w, method, covered_only=True, mapping=mapping))
        elif not any(mapping[w].values()):
            acc.append(0.0)
    return mean(acc), mean(cov)


def results_table(compare, mapping):
    """One row per (name, predictions, method), with accuracy and coverage for acronyms and terms."""
    rows = []
    for name, predictions, method in compare:
        line = {"method": name}
        for group, words in [("acronyms", ALL_ACRONYMS), ("terms", ALL_TERMS)]:
            line[f"{group} acc"], line[f"{group} cov"] = group_scores(predictions, mapping, words, method)
        rows.append(line)
    return pd.DataFrame(rows).set_index("method").round(3)


def step_results():
    """Print and save the result tables of Task 4 and Task 6 (also with partial predictions)."""
    mapping = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    n_covered = sum(any(m.values()) for m in mapping.values())

    if TASK4_FILE.exists():
        pred4 = json.loads(TASK4_FILE.read_text(encoding="utf-8"))
        table4 = results_table([(m, pred4, m) for m in ALL_METHODS], mapping)
        print(f"\n=== Task 4   ({len(pred4)} of {n_covered} covered words done)")
        print(table4.to_string())
        table4.to_csv(RESULTS / "task4_results.csv")

    if TASK6_FILE.exists() and TASK4_FILE.exists():
        pred6 = json.loads(TASK6_FILE.read_text(encoding="utf-8"))
        compare = [("Wu-Palmer (pywsd, Task 4)", pred4, "Wu-Palmer")]
        compare += [(f"alpha = {a}", pred6, str(a)) for a in ALPHAS]
        table6 = results_table(compare, mapping)
        print(f"\n=== Task 6   ({len(pred6)} of {n_covered} covered words done)")
        print(table6.to_string())
        table6.to_csv(RESULTS / "task6_results.csv")

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "info"
    if step == "info":
        step_info()
    elif step == "umls":
        step_umls()
    elif step == "mapping":
        min_overlap = int(sys.argv[2]) if len(sys.argv) > 2 else MIN_OVERLAP
        step_mapping(min_overlap)
    elif step == "task4":
        step_task4()
    elif step == "task6":
        step_task6()
    elif step == "results":
        step_results()
    else:
        print("Unknown step:", step)