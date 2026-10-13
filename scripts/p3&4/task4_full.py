"""
Task 4 on the FULL MSH-WSD dataset (all 203 words), with an automatic CUI -> WordNet mapping.

Run the steps from the terminal, in the scripts/ folder, in this order:

    python task4_full.py umls                     # 1) download UMLS names and definitions of every CUI (asks the API key)
    python task4_full.py mapping                  # 2) automatic mapping + validation on our 39 manual words
    python task4_full.py run --methods fast       # 3a) Lesk methods + baseline (~30 min)
    python task4_full.py run --methods similarity # 3b) Wu-Palmer + Lin (hours: run it at night)
    python task4_full.py results                  # 4) tables of the results

Every step saves its output in a file, so a step can be interrupted and restarted:
it continues from where it stopped.
"""

import argparse
import getpass
import html
import json
import random
import re
import time
from pathlib import Path

import requests
from nltk.corpus import stopwords
from nltk.corpus import wordnet as nwn          # NLTK WordNet 3.0 (used by our mapping)
from nltk.wsd import lesk as nltk_lesk

from pywsd.lesk import original_lesk, simple_lesk, adapted_lesk, cosine_lesk
from pywsd.similarity import max_similarity
from pywsd.baseline import max_lemma_count
from pywsd.utils import lemmatize
from pywsd._wordnet import wn as pwn            # Open English WordNet 2024 (used by pywsd)
import wn.compat.sensekey as sensekey

# ---------------------------------------------------------------- parameters and files
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "MSHCorpus" / "MSHCorpus"
SYNONYMS_FILE = HERE.parent / "data" / "umls_cui_info.json"     # UMLS content: must NOT go on GitHub
MANUAL_FILE = HERE / "manual_mapping.json"                      # the MAPPING of task_4.ipynb
AUTO_FILE = HERE / "auto_mapping.json"
WINDOW = 10                                                     # context words on each side of the target
METHOD_GROUPS = {"fast": 10, "similarity": 5}                   # abstracts per sense for each group

STOP = set(stopwords.words("english"))


# ---------------------------------------------------------------- data (same as task 4)
def load_concept_map():
    """{word: {"M1": CUI, "M2": CUI, ...}} from benchmark_mesh.txt."""
    concept_map = {}
    with open(DATA_DIR / "benchmark_mesh.txt", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                concept_map[parts[0]] = {f"M{i}": cui for i, cui in enumerate(parts[1:], start=1)}
    return concept_map


CONCEPT_MAP = load_concept_map()


def load_instances(word):
    """All the abstracts of one word, with the tagged text and the correct CUI."""
    lines = (DATA_DIR / f"{word}_pmids_tagged.arff").read_text(encoding="iso-8859-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("@DATA")) + 1
    instances = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        pmid, rest = line.split(",", 1)
        text, label = rest.rsplit(",", 1)
        text = html.unescape(text.strip('"'))
        instances.append({"pmid": pmid, "tagged": text, "cui": CONCEPT_MAP[word][label.strip()]})
    return instances


def context_window(tagged_text, n=WINDOW):
    """The n words before the target, the target itself and the n words after it."""
    if "<e>" not in tagged_text:
        return re.sub(r"</?e>", "", tagged_text)
    before, rest = tagged_text.split("<e>", 1)
    target, after = rest.split("</e>", 1)
    words_before = re.findall(r"\w+", before)[-n:]
    words_after = re.findall(r"\w+", re.sub(r"</?e>", "", after))[:n]
    return " ".join(words_before + [target] + words_after)


def sample_instances(word, n, seed=0):
    """Random sample of n abstracts for each sense of the word."""
    rng = random.Random(seed)
    by_sense = {}
    for inst in load_instances(word):
        by_sense.setdefault(inst["cui"], []).append(inst)
    sample = []
    for cui, insts in sorted(by_sense.items()):
        sample += rng.sample(insts, min(n, len(insts)))
    return sample


# ---------------------------------------------------------------- acronyms and terms (same as task 4)
MANUAL_ACRONYMS = {"Ca", "Ala", "Epi", "Fe", "dC", "pI", "Staph", "Strep"}
MANUAL_TERMS = {"ERUPTION", "LABOR", "NEUROFIBROMATOSIS", "THYMUS", "Gamma-Interferon"}
EXCLUDED = {"EGG", "STEM", "Orf"}


def is_acronym(word):
    if " " in word:
        return False
    return sum(c.isupper() for c in word) >= 2 or any(c.isdigit() for c in word)


ALL_WORDS = sorted(w for w in CONCEPT_MAP if w not in EXCLUDED)
ACRONYMS = [w for w in ALL_WORDS if (is_acronym(w) or w in MANUAL_ACRONYMS) and w not in MANUAL_TERMS]
TERMS = [w for w in ALL_WORDS if w not in ACRONYMS]


def wordnet_senses(word):
    """NLTK WordNet senses of an MSH-WSD word (same as task 2)."""
    w = word.lower().replace(" ", "_")
    senses = nwn.synsets(w)
    if not senses:
        base = nwn.morphy(w)
        if base:
            senses = nwn.synsets(base)
    return senses


# ---------------------------------------------------------------- step 1: UMLS names and definitions
UTS = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI"


def umls_info(cui, api_key):
    """English names (preferred name + synonyms) and definitions of a CUI in UMLS, lowercase."""
    info = {"names": [], "definitions": []}
    r = requests.get(f"{UTS}/{cui}/atoms", params={"apiKey": api_key, "language": "ENG", "pageSize": 200}, timeout=30)
    if r.ok:                         # some 2009 CUIs no longer exist in the current UMLS
        info["names"] = sorted({atom["name"].lower() for atom in r.json()["result"]})
    r = requests.get(f"{UTS}/{cui}/definitions", params={"apiKey": api_key, "pageSize": 50}, timeout=30)
    if r.ok:                         # many CUIs have no definition
        info["definitions"] = [d["value"].lower() for d in r.json()["result"]]
    return info


def cmd_umls(args):
    umls = json.loads(SYNONYMS_FILE.read_text(encoding="utf-8")) if SYNONYMS_FILE.exists() else {}
    all_cuis = sorted({cui for senses in CONCEPT_MAP.values() for cui in senses.values()})
    todo = [c for c in all_cuis if c not in umls]
    print(f"{len(all_cuis)} CUIs, {len(todo)} still to download")
    if todo:
        api_key = getpass.getpass("UMLS API key: ")
    for i, cui in enumerate(todo, 1):
        umls[cui] = umls_info(cui, api_key)
        if i % 20 == 0 or i == len(todo):            # save every 20 CUIs
            SYNONYMS_FILE.write_text(json.dumps(umls, indent=1), encoding="utf-8")
            print(f"   {i}/{len(todo)}")
    print(f"{sum(1 for c in all_cuis if not umls[c]['names'])} CUIs not found in the current UMLS, "
          f"{sum(1 for c in all_cuis if not umls[c]['definitions'])} without definition")
    print("Example C0015422:", umls.get("C0015422"))


# ---------------------------------------------------------------- step 2: automatic mapping
def normalize(name):
    """'Eyelashes' -> 'eyelash', 'breastfeeding' -> 'breastfeed', 'adenosine_diphosphate' -> 'adenosine diphosphate'."""
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return " ".join(nwn.morphy(t, "n") or nwn.morphy(t, "v") or t for t in tokens)


def content_words(text, excluded):
    """Normalized words of a text, without stopwords, short words and the target word itself."""
    return {w for w in normalize(text).split() if w not in STOP and len(w) > 2 and w not in excluded}


def auto_mapping_word(word, umls, min_overlap=2):
    """Automatic CUI -> synsets mapping of one word.

    Rule 1 (name): a synset goes to a CUI if one of its lemmas (not the target word itself)
                   is equal to a UMLS name of the CUI, e.g. eyelash.n.01 -> 'eyelash'.
    Rule 2 (words): otherwise, we count the words shared by the synset (definition, lemmas,
                   hypernyms) and the CUI (UMLS names and definitions), and take the CUI with
                   most shared words, if they are at least min_overlap.
    In both rules the winner must be unique, so each synset goes to at most one CUI.
    """
    cuis = list(CONCEPT_MAP[word].values())
    target = set(normalize(word).split())                     # the target word gives no information
    empty = {"names": [], "definitions": []}
    names = {c: {normalize(n) for n in umls.get(c, empty)["names"]} for c in cuis}
    words = {c: content_words(" ".join(umls.get(c, empty)["names"] + umls.get(c, empty)["definitions"]), target)
             for c in cuis}

    mapping = {c: [] for c in cuis}
    for s in wordnet_senses(word):
        lemmas = {normalize(l) for l in s.lemma_names()} - {normalize(word)}
        matches = [c for c in cuis if lemmas & names[c]]          # rule 1
        if len(matches) != 1:
            bag = content_words(s.definition() + " " + " ".join(s.lemma_names()), target)
            for h in s.hypernyms():
                bag |= content_words(" ".join(h.lemma_names()), target)
            scores = {c: len(bag & words[c]) for c in cuis}       # rule 2
            best = max(scores.values())
            matches = [c for c in cuis if scores[c] == best] if best >= min_overlap else []
        if len(matches) == 1:
            mapping[matches[0]].append(s.name())
    return mapping


def validate(auto, manual):
    """Compare the automatic mapping with the manual one, as (CUI, synset) pairs."""
    auto_pairs = {(w, c, s) for w in manual for c, ss in auto[w].items() for s in ss}
    manual_pairs = {(w, c, s) for w in manual for c, ss in manual[w].items() for s in ss}
    correct = auto_pairs & manual_pairs
    precision = len(correct) / len(auto_pairs) if auto_pairs else float("nan")
    recall = len(correct) / len(manual_pairs) if manual_pairs else float("nan")
    # same decision about "this CUI exists / does not exist in WordNet"
    same_coverage = [bool(auto[w][c]) == bool(manual[w][c]) for w in manual for c in manual[w]]
    return precision, recall, sum(same_coverage) / len(same_coverage), auto_pairs - manual_pairs, manual_pairs - auto_pairs


def cmd_mapping(args):
    umls = json.loads(SYNONYMS_FILE.read_text(encoding="utf-8"))
    auto = {w: auto_mapping_word(w, umls, args.min_overlap) for w in ALL_WORDS}
    AUTO_FILE.write_text(json.dumps(auto, indent=1), encoding="utf-8")

    n_cuis = sum(len(m) for m in auto.values())
    n_covered = sum(bool(ss) for m in auto.values() for ss in m.values())
    print(f"{len(auto)} words, {n_cuis} CUIs, {n_covered} CUIs with at least one synset -> {AUTO_FILE.name}")

    if MANUAL_FILE.exists():
        manual = json.loads(MANUAL_FILE.read_text(encoding="utf-8"))
        p, r, cov, wrong, missing = validate(auto, manual)
        print(f"\nValidation on the {len(manual)} manual words:")
        print(f"   precision {p:.2f}   (automatic pairs that are also in the manual mapping)")
        print(f"   recall    {r:.2f}   (manual pairs found by the automatic mapping)")
        print(f"   coverage agreement {cov:.2f}   (same decision 'in WordNet / not in WordNet')")
        print("   wrong  :", sorted(wrong))
        print("   missing:", sorted(missing))
    else:
        print(f"\n{MANUAL_FILE.name} not found: no validation")


# ---------------------------------------------------------------- step 3: running the methods (same as task 4)
_get_key = sensekey.sense_key_getter("oewn:2024")


def to_nltk(oewn_synset):
    """Translate a pywsd (Open English WordNet 2024) synset into its NLTK (WordNet 3.0) name."""
    if oewn_synset is None:
        return None
    for sense in oewn_synset.senses():
        key = _get_key(sense)
        if key:
            try:
                return nwn.lemma_from_key(key).synset().name()
            except Exception:
                continue
    return None


_form_cache = {}


def pywsd_form(word, mapping):
    """Spelling of the word that lets pywsd see most of the senses used in the mapping."""
    if word not in _form_cache:
        mapped = {s for synsets in mapping[word].values() for s in synsets}
        best_form, best_score = word, (-1, -1)
        for form in dict.fromkeys([word, word.lower(), word.capitalize(), word.upper()]):
            candidates = {to_nltk(s) for s in pwn.synsets(lemmatize(form))}
            score = (len(candidates & mapped), len(candidates))
            if score > best_score:
                best_form, best_score = form, score
        _form_cache[word] = best_form
    return _form_cache[word]


def resnik_fixed(context, word):
    """Resnik with the ranking sorted correctly (highest score = most similar)."""
    ranking = max_similarity(context, word, option="res", best=False)
    return max(ranking, key=lambda x: x[0])[1] if ranking else None


METHODS = {
    "fast": {
        "Original Lesk":   lambda ctx, w: original_lesk(ctx, w),
        "Simple Lesk":     lambda ctx, w: simple_lesk(ctx, w),
        "Adapted Lesk":    lambda ctx, w: adapted_lesk(ctx, w),
        "Cosine Lesk":     lambda ctx, w: cosine_lesk(ctx, w),
        "Max lemma count": lambda ctx, w: max_lemma_count(w),
    },
    "similarity": {
        "Wu-Palmer":       lambda ctx, w: max_similarity(ctx, w, option="wup"),
        "Lin":             lambda ctx, w: max_similarity(ctx, w, option="lin"),
    },
}


def predict(group, method, context, word, mapping):
    """Answer of one method as an NLTK synset name (None if no answer)."""
    try:
        if method == "NLTK Lesk":
            senses = wordnet_senses(word)
            answer = nltk_lesk(context.lower().split(), word.lower(), synsets=senses) if senses else None
            return answer.name() if answer else None
        return to_nltk(METHODS[group][method](context, pywsd_form(word, mapping)))
    except Exception:
        return None


def predictions_file(group):
    return HERE / f"task4_full_{group}.json"


def cmd_run(args):
    group = args.methods
    n = METHOD_GROUPS[group]
    methods = (["NLTK Lesk"] if group == "fast" else []) + list(METHODS[group])
    mapping = json.loads(AUTO_FILE.read_text(encoding="utf-8"))
    out = predictions_file(group)
    predictions = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    words = [w for w in ALL_WORDS if w not in predictions]
    print(f"{group}: {methods}, {n} abstracts per sense, {len(words)} words to do")

    start = time.time()
    for i, word in enumerate(words, 1):
        t = time.time()
        covered = any(mapping[word].values())     # no sense in WordNet: every answer is wrong anyway
        rows = []
        for inst in sample_instances(word, n):
            ctx = context_window(inst["tagged"])
            answers = {m: predict(group, m, ctx, word, mapping) if covered else None for m in methods}
            rows.append({"cui": inst["cui"], "answers": answers})
        predictions[word] = rows
        out.write_text(json.dumps(predictions), encoding="utf-8")
        elapsed = (time.time() - start) / 60
        print(f"[{i:3}/{len(words)}] {word:20} {len(rows):3} abstracts {time.time() - t:6.1f} s   "
              f"(total {elapsed:.0f} min)")
    print("Done.")


# ---------------------------------------------------------------- step 4: results
def accuracy(rows, word, method, mapping, covered_only=False):
    if covered_only:
        rows = [r for r in rows if mapping[word][r["cui"]]]
    if not rows:
        return None
    correct = [r["answers"][method] in mapping[word][r["cui"]] for r in rows]
    return sum(correct) / len(correct)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def cmd_results(args):
    import pandas as pd
    mapping = json.loads(AUTO_FILE.read_text(encoding="utf-8"))
    table = []
    for group in METHOD_GROUPS:
        if not predictions_file(group).exists():
            continue
        predictions = json.loads(predictions_file(group).read_text(encoding="utf-8"))
        methods = next(iter(predictions.values()))[0]["answers"].keys()
        for method in methods:
            line = {"method": method}
            for group_name, words in [("acronyms", ACRONYMS), ("terms", TERMS)]:
                words = [w for w in words if w in predictions]
                line[f"{group_name} acc"] = mean(accuracy(predictions[w], w, method, mapping) for w in words)
                line[f"{group_name} cov"] = mean(accuracy(predictions[w], w, method, mapping, True) for w in words)
                line[f"{group_name} words"] = len(words)
            table.append(line)
    results = pd.DataFrame(table).set_index("method").round(3)
    print(results.to_string())
    results.to_csv(HERE / "task4_full_results.csv")
    print("\n-> task4_full_results.csv")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["umls", "mapping", "run", "results"])
    parser.add_argument("--methods", choices=list(METHOD_GROUPS), default="fast")
    parser.add_argument("--min-overlap", type=int, default=2, help="rule 2: minimum shared words")
    args = parser.parse_args()
    {"umls": cmd_umls, "mapping": cmd_mapping, "run": cmd_run, "results": cmd_results}[args.step](args)
