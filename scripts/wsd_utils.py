"""Shared functions for the WSD tasks (Task 2, 3, 4 and 6) on the MSH-WSD dataset."""

import html
import re
from pathlib import Path
from nltk.corpus import wordnet as wn   
import random

from pywsd._wordnet import wn as pwn       
from pywsd.utils import lemmatize
import wn.compat.sensekey as sensekey


# ---------------------------------------------------------------- reading the dataset -------------------------------------------------------------
# MSH-WSD has two kinds of files:
# - benchmark_mesh.txt: one line per ambiguous word, with the CUIs of all its senses;
# - <word>_pmids_tagged.arff: the abstracts of one word, with the word marked as <e>word</e>
#   and the correct sense written as a label (M1 = first CUI, M2 = second CUI, ...).

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "MSHCorpus"


def load_concept_map():
    """Read benchmark_mesh.txt: word -> {"M1": CUI, "M2": CUI, ...}."""
    concept_map = {}
    with open(DATA_DIR / "benchmark_mesh.txt", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            concept_map[parts[0]] = {f"M{i}": cui for i, cui in enumerate(parts[1:], start=1)}
    return concept_map

CONCEPT_MAP = load_concept_map()

def load_instances(word):
    """Read all the abstracts of one ambiguous word."""
    path = DATA_DIR / f"{word}_pmids_tagged.arff"
    lines = path.read_text(encoding="iso-8859-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("@DATA")) + 1   # abstracts start after "@DATA"

    instances = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        pmid, rest = line.split(",", 1)            # text before the FIRST comma = PMID
        text, label = rest.rsplit(",", 1)          # text after the LAST comma = label (M1, M2...)
        text = html.unescape(text.strip('"'))      # remove quotes, convert &lt; -> <
        marked = re.search(r"<e>(.+?)</e>", text)
        instances.append({
            "word": word,
            "pmid": pmid,
            "target": marked.group(1) if marked else word,   # the marked word, e.g. "cements"
            "tagged": text,                                  # text with the <e> </e> tags
            "context": re.sub(r"</?e>", "", text),           # clean text
            "label": label.strip(),                          # e.g. "M1"
            "cui": CONCEPT_MAP[word][label.strip()],         # correct sense, e.g. "C0002520"
        })
    return instances

# ---------------------------------------------------------------- acronyms and terms -------------------------------------------------------------
# The dataset does not say which words are acronyms, so we use a simple rule
# (2+ capital letters or a digit) and correct its mistakes by hand.

def is_acronym(word):
    """An acronym has at least 2 capital letters or a digit, and no spaces."""
    if " " in word:
        return False
    return sum(c.isupper() for c in word) >= 2 or any(c.isdigit() for c in word)


# Manual corrections of the rule (checked by hand)
MANUAL_ACRONYMS = {"Ca", "Ala", "Epi", "Fe", "dC", "pI", "Staph", "Strep"}
MANUAL_TERMS    = {"ERUPTION", "LABOR", "NEUROFIBROMATOSIS", "THYMUS", "Gamma-Interferon"}
EXCLUDED        = {"EGG", "STEM", "Orf"}          # can be both a term and an acronym

ALL_WORDS    = sorted(w for w in CONCEPT_MAP if w not in EXCLUDED)
ALL_ACRONYMS = [w for w in ALL_WORDS if (is_acronym(w) or w in MANUAL_ACRONYMS) and w not in MANUAL_TERMS]
ALL_TERMS    = [w for w in ALL_WORDS if w not in ALL_ACRONYMS]

# ---------------------------------------------------------------- WordNet senses -------------------------------------------------------------
# Lesk can only choose among the senses listed in WordNet (NLTK, WordNet 3.0).
# WordNet uses lowercase words with underscores, and base forms instead of plurals.

def wordnet_senses(word):
    """NLTK WordNet senses of an MSH-WSD word ("B-Cell Leukemia" -> b-cell_leukemia, "Follicles" -> follicle)."""
    w = word.lower().replace(" ", "_")
    senses = wn.synsets(w)
    if not senses:                 # try the base form, e.g. plural -> singular
        base = wn.morphy(w)
        if base:
            senses = wn.synsets(base)
    return senses


# ---------------------------------------------------------------- CUI -> WordNet mapping -------------------------------------------------------------
# Built by hand for the 39 words of our two selections.
# For each word: CUI -> list of WordNet synsets with the same meaning.
# Rules: [] if WordNet has no equivalent sense; each synset goes to at most one CUI.

MAPPING = {
    # ---------------- acronyms (selection 1) ----------------
    "ADP":   {"C0001459": ["adenosine_diphosphate.n.01"],      # Adenosine Diphosphate
              "C0004374": ["automatic_data_processing.n.01"]}, # Automatic Data Processing
    "CAD":   {"C0011905": [],                                  # Computer Assisted Diagnosis (WN has only "design")
              "C1956346": []},                                 # Coronary Artery Disease
    "CAM":   {"C0007578": [],                                  # Cell Adhesion Molecules
              "C0178551": []},                                 # chorioallantoic membrane
    "CDA":   {"C0092801": [],                                  # cladribine
              "C0002876": []},                                 # Congenital dyserythropoietic anemia
    "Ca":    {"C0006675": ["calcium.n.01"],                    # calcium
              "C0006754": ["california.n.01"],                 # California
              "C0019564": [],                                  # Hippocampus
              "C0006823": []},                                 # Canada
    "DDD":   {"C0026256": [],                                  # mitotane
              "C0011037": []},                                 # Dichlorodiphenyldichloroethane
    "DON":   {"C0028652": [],                                  # Nurse Administrator
              "C0012020": []},                                 # Diazooxonorleucine
    "SARS":  {"C1175175": ["severe_acute_respiratory_syndrome.n.01"],  # the disease
              "C1175743": []},                                 # the coronavirus
    "Staph": {"C0038160": [],                                  # Staphylococcal Infections
              "C0038170": ["staphylococcus.n.01"]},            # Genus staphylococcus
    "TNT":   {"C0077404": [],                                  # Troponin T
              "C0041070": ["tnt.n.01"]},                       # Trinitrotoluene
    # ---------------- acronyms (selection 2) ----------------
    "ALS":   {"C0003372": [],                                         # Antilymphocyte Serum
              "C0002736": ["amyotrophic_lateral_sclerosis.n.01"]},    # Amyotrophic Lateral Sclerosis
    "ANA":   {"C0002463": [],                                         # American Nurses' Association
              "C0003243": []},                                        # Antinuclear Antibodies
    "CI":    {"C0022326": [],                                         # Ivory Coast
              "C0008107": []},                                        # Chile
    "CLS":   {"C0265252": [],                                         # Coffin-Lowry syndrome
              "C0343084": []},                                        # Capillary Leak Syndrome
    "FAS":   {"C0015683": [],                                         # Fatty-acid synthase
              "C0015923": ["fetal_alcohol_syndrome.n.01"]},           # Fetal Alcohol Syndrome
    "GAG":   {"C0017346": [],                                         # GAG Gene
              "C0017973": []},                                        # Glycosaminoglycans
    "MRS":   {"C0025235": [],                                         # Melkersson-Rosenthal Syndrome
              "C0024487": []},                                        # Magnetic Resonance Spectroscopy
    "PVC":   {"C0151636": ["premature_ventricular_contraction.n.01"], # Premature ventricular contractions
              "C0032624": ["polyvinyl_chloride.n.01"]},               # Polyvinyl Chloride
    "RA":    {"C0002893": [],                                         # Refractory anemias
              "C0034625": ["radium.n.01"],                            # radium
              "C0003873": []},                                        # Rheumatoid Arthritis
    "US":    {"C0041703": ["united_states.n.01"],                     # United States
              "C0041618": []},                                        # Ultrasonography
    # ---------------- regular terms (selection 1) ----------------
    "B-Cell Leukemia": {"C2004493": [],                        # Leukemia, B-Cell
                        "C0023434": []},                       # Chronic Lymphocytic Leukemia
    "Borrelia":    {"C0024198": [],                            # Lyme Disease
                    "C0006033": ["borrelia.n.01"]},            # Borrelia (bacterium)
    "Cilia":       {"C0008778": ["cilium.n.01"],               # Cilia (cell projection) - also in selection 2
                    "C0015422": ["eyelash.n.01"]},             # Eyelash
    "Follicles":   {"C0221971": [],                            # Hair follicle
                    "C0018120": ["follicle.n.01"]},            # Ovarian Follicle (closest to WN gloss)
    "Ganglion":    {"C0017067": ["ganglion.n.01"],             # Ganglia (neural)
                    "C1258666": []},                           # Myxoid cyst
    "Nurse":       {"C0028661": ["nurse.n.01", "nurse.v.03"],  # Nurses (the profession)
                    "C0006147": ["breastfeed.v.01"]},          # Breast Feeding
    "Pneumocystis": {"C0032305": [],                           # (retired CUI, name not available)
                     "C0597258": []},                          # Genus Pneumocystis
    "Respiration": {"C0035203": ["respiration.n.02", "breathing.n.01"],  # breathing
                    "C0282636": ["respiration.n.01"]},         # Cell Respiration (metabolic)
    "Sodium":      {"C0037473": ["sodium.n.01"],               # sodium (element)
                    "C0037570": []},                           # Sodium, Dietary
    "Synapsis":    {"C0039062": [],                            # Synapses (neuron junction)
                    "C0598501": ["synapsis.n.01"]},            # Chromosome Pairing
    # ---------------- regular terms (selection 2) ----------------
    "Callus":      {"C0006767": ["callus.n.02"],                     # Bone callus
                    "C0376154": ["callosity.n.01"]},                 # Skin callus
    "Cell":        {"C0007634": ["cell.n.02"],                       # Cells (biology)
                    "C1136359": ["cellular_telephone.n.01"]},        # Cellular Phone
    "Cement":      {"C1706094": ["cement.n.04", "cement.n.03"],      # Dental Cement Material
                    "C0011343": ["cementum.n.01"]},                  # Dental Cementum
    "ERUPTION":    {"C0015230": ["eruption.n.02"],                   # Exanthema (skin rash)
                    "C1533692": ["eruption.n.06"]},                  # Tooth eruption
    "Exercises":   {"C0452240": ["exercise.n.03"],                   # Physical therapy exercises
                    "C0015259": ["exercise.n.01"]},                  # Exercise
    "Medullary":   {"C0001629": ["medullary.a.03"],                  # Adrenal Medulla
                    "C0025148": ["medullary.a.02"]},                 # Medulla Oblongata
    "Milk":        {"C0026131": ["milk.n.01"],                       # Milk (body substance)
                    "C0026140": ["milk.n.02"]},                      # Milk, Human
    "Phosphorus":  {"C0080014": [],                                  # Dietary Phosphorus
                    "C0031705": ["phosphorus.n.01"]},                # phosphorus (element)
    "Ventricles":  {"C0018827": ["ventricle.n.02"],                  # Heart Ventricle
                    "C0007799": ["ventricle.n.01"]},                 # Cerebral Ventricles
}

# ---------------------------------------------------------------- pywsd (Tasks 4 and 6)
# pywsd uses Open English WordNet 2024, while our MAPPING uses the NLTK names (WordNet 3.0).
# We translate the pywsd answers with the sense key, which identifies the same sense in both versions.

get_key = sensekey.sense_key_getter("oewn:2024")   # pywsd sense -> sense key


def to_nltk(pywsd_synset):
    """Translate a pywsd synset (e.g. oewn-01460746-n) into its NLTK name (e.g. cilium.n.01)."""
    if pywsd_synset is None:
        return None
    for sense in pywsd_synset.senses():
        key = get_key(sense)
        if key:
            try:
                return wn.lemma_from_key(key).synset().name()
            except Exception:
                continue
    return None


def pywsd_form(word):
    """Spelling of the word that lets pywsd see most of the senses used in MAPPING
    (pywsd lemmatizes the word and is case-sensitive: "ALS" -> "al", "US" -> "u")."""
    mapped = {s for synsets in MAPPING[word].values() for s in synsets}
    best_form, best_score = word, (-1, -1)
    for form in dict.fromkeys([word, word.lower(), word.capitalize(), word.upper()]):
        found = {to_nltk(s) for s in pwn.synsets(lemmatize(form))}
        score = (len(found & mapped), len(found))
        if score > best_score:
            best_form, best_score = form, score
    return best_form


# ---------------------------------------------------------------- context window and sample (Tasks 4 and 6)
# The similarity methods are slow, so we use only the words around the target
# and a balanced random sample of abstracts (same number for each sense).

WINDOW = 10         # context words on each side of the target
N_PER_SENSE = 10    # abstracts sampled for each sense of each word


def context_window(tagged_text, n=WINDOW):
    """The n words before the target, the target itself and the n words after it."""
    if "<e>" not in tagged_text:
        return re.sub(r"</?e>", "", tagged_text)
    before, rest = tagged_text.split("<e>", 1)
    target, after = rest.split("</e>", 1)
    words_before = re.findall(r"\w+", before)[-n:]
    words_after = re.findall(r"\w+", re.sub(r"</?e>", "", after))[:n]
    return " ".join(words_before + [target] + words_after)


def sample_instances(word, n=N_PER_SENSE, seed=0):
    """Random sample of n abstracts for each sense (CUI) of the word (fixed seed, reproducible)."""
    rng = random.Random(seed)
    by_sense = {}
    for inst in load_instances(word):
        by_sense.setdefault(inst["cui"], []).append(inst)
    sample = []
    for cui in sorted(by_sense):
        sample += rng.sample(by_sense[cui], min(n, len(by_sense[cui])))
    return sample


# ---------------------------------------------------------------- evaluation helpers -------------------------------------------------------------

def mean(values):
    """Average that ignores None (e.g. words with no covered sense)."""
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")

def accuracy(rows, word, method, covered_only=False):
    """Share of correct answers of one method on the saved predictions of one word.
    With covered_only, only the abstracts whose correct sense exists in WordNet."""
    if covered_only:
        rows = [r for r in rows if MAPPING[word][r["cui"]]]
    if not rows:
        return None
    correct = [r["answers"][method] in MAPPING[word][r["cui"]] for r in rows]
    return sum(correct) / len(correct)