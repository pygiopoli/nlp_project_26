# Project 31 — Word Sense Disambiguation in Medical Context

> A comparative study of knowledge-based Word Sense Disambiguation (WSD) algorithms for biomedical and clinical text, with a proposed hybrid semantic–phonetic approach.

**Status:** In development · **Language:** Python · **Domain:** Biomedical Natural Language Processing

## Overview

Medical language contains many ambiguous terms and acronyms. The same expression may refer to different biomedical concepts depending on its surrounding context. **Word Sense Disambiguation (WSD)** identifies the intended meaning of a target word or expression in a sentence or document.

This project investigates classical dictionary-based and semantic-similarity approaches to WSD, evaluates them on the **MSH-WSD** biomedical benchmark, and explores whether adding **phonetic similarity** to a Wu–Palmer-based method can improve disambiguation.

The project covers eight tasks: (1) a baseline NLTK Lesk example; (2) MSH-WSD evaluation on sampled terms and acronyms; (3) a comparison of `pywsd` algorithms; (4) an extended evaluation on the dataset; (5) phonetic-distance analysis; (6) a new combined semantic–phonetic scorer; (7) an interactive interface; and (8) a literature review and critical discussion.

## Research questions

1. How reliable is the standard Lesk algorithm when applied to a simple medically relevant sentence?
2. Do knowledge-based WSD methods perform differently on biomedical **regular terms** and **acronyms**?
3. How do overlap-based, taxonomy-based, information-content-based, and lemma-frequency-based methods compare?
4. Can phonetic information complement Wu–Palmer semantic similarity, and under which conditions is it useful?
5. What limitations arise when general-purpose WordNet methods are evaluated against biomedical concept identifiers?

## Dataset: MSH-WSD

The **MSH-WSD** benchmark was introduced by Jimeno-Yepes, McInnes, and Aronson (2011). It contains **203 ambiguous expressions**: **106 abbreviations**, **88 regular terms**, and **9 mixed cases**. Contexts are drawn from MEDLINE, and senses are labeled with **UMLS Concept Unique Identifiers (CUIs)**.

For the sampled experiment, we will randomly choose **10 acronyms** and **10 regular terms**, using a fixed random seed. The full evaluation will report results separately for **all 106 abbreviation entries** and **all 88 regular-term entries**. Mixed entries will be analyzed separately or explicitly excluded from the two headline averages.

**Data-access note:** The [Yarn repository](https://github.com/clips/yarn) referenced in the assignment provides biomedical concept-disambiguation software and example input formats; **it does not bundle the full MSH-WSD dataset or the required UMLS resources**. Dataset acquisition, permissions, and any UMLS licensing requirements must therefore be handled separately. Do not commit restricted datasets or licensed terminology resources to this repository.

### Sense alignment (important)

NLTK and `pywsd` generally produce **WordNet synsets**, while MSH-WSD labels refer to **UMLS CUIs**. These are different sense inventories: their identifiers cannot be compared directly. Before calculating accuracy, the evaluation pipeline must define and document a defensible mapping between WordNet predictions and MSH-WSD candidate concepts, or instead use a concept-aware adaptation that scores descriptions associated with the candidate CUIs. **Unmapped and unscorable cases must be counted and reported**, not silently treated as correct.

## Methods and experiments

### 1. Baseline: standard Lesk in NLTK

Example sentence:

> I have been prescribed two important drugs today during my visit to clinic.

The target is **`drug`** (with the plural surface form **`drugs`** in the sentence). We will tokenize the sentence, apply the standard NLTK Lesk implementation, and inspect the selected WordNet sense, its definition, and its overlap with the sentence context.

Lesk estimates the intended sense by comparing the words in a candidate sense's gloss with the surrounding sentence. Its strengths are simplicity and interpretability; its weaknesses include short glosses, vocabulary mismatch, limited medical coverage, and sensitivity to preprocessing. The result is **not assumed correct in advance**: it will be compared with the intended medication-related sense.

### 2. Sampled MSH-WSD evaluation

- Fix and record a random seed for reproducibility.
- Select 10 abbreviation entries and 10 regular-term entries (not just 20 individual sentences).
- Evaluate each available labeled context for those entries using a Lesk-based baseline.
- Apply a consistent preprocessing policy and a documented sense-alignment strategy.
- Report the accuracy for each entry and the **macro-average accuracy** separately for abbreviations and regular terms.

### 3. Comparison of existing WSD algorithms

Using [pywsd](https://github.com/alvations/pywsd), and small documented wrappers where necessary, we will run the following methods on a sentence containing a target word:

| Method | Main idea |
|---|---|
| Original Lesk | Overlap between context and dictionary gloss |
| Adapted Lesk | Expanded overlap using related glosses |
| Simple/extended Lesk with hypernyms and hyponyms | Include broader and narrower concepts |
| Cosine Lesk | Vector-based similarity between context and sense descriptions |
| Path similarity | WordNet taxonomy distance |
| Information-content similarity | Taxonomy plus corpus-derived information content |
| Highest lemma count | Most-frequent-sense proxy based on lemma counts |
| Wu–Palmer similarity | Similarity based on depth and lowest common subsumer in a taxonomy |

*Note:* Method availability, exact functions, and version-specific behavior will be verified against the installed `pywsd` version. Not every method listed above necessarily uses the same interface or qualifies as a Lesk variant.

### 4. Full dataset benchmark

Run each selected method over all evaluable contexts for the 106 abbreviation and 88 regular-term entries. Record prediction, gold CUI, match/mismatch, mapping status, and processing time where appropriate. Report the same metrics and evaluation protocol across algorithms, including coverage and missing predictions.

### 5. Phonetic similarity

We will use the Python **`fuzzy`** library to derive phonetic encodings (for example, Double Metaphone, subject to library compatibility) and an edit-distance function to compare the resulting encodings.

For words or encodings `a` and `b`, one possible normalized similarity is:

\[
P(a,b) = 1 - \frac{d(e(a), e(b))}{\max(|e(a)|, |e(b)|)}
\]

where `e` denotes a phonetic encoding and `d` denotes Levenshtein edit distance. If both encodings are empty, the implementation must define an explicit fallback rather than divide by zero.

The experiment will create a **symmetric pairwise phonetic-distance matrix** for the words in the example sentence in Task 1, accompanied by a heatmap. The matrix must state whether it displays **raw edit distance** (lower means more similar) or **normalized similarity** (higher means more similar).

### 6. Proposed hybrid: Wu–Palmer + phonetic matching

We propose combining a normalized Wu–Palmer semantic score `W` and a normalized phonetic score `P` through the required convex combination:

\[
S_{hybrid}(c,s) = 0.5\,W(c,s) + 0.5\,P(c,s)
\]

where `c` is the observed context and `s` is a candidate sense. Both components will be normalized to **[0, 1]**. The system will choose the candidate with the highest combined score.

**Design requirement:** Wu–Palmer operates on **WordNet synsets**, whereas phonetic encoders operate on **strings**. The implementation must explicitly define how the context is converted to comparison synsets, how each candidate sense is represented as text, how the two scores are aggregated, and how missing semantic/phonetic values are handled. For MSH-WSD, a CUI-to-WordNet alignment or an explicitly documented biomedical concept-scoring adaptation is needed. The 0.5/0.5 weights are specified by the assignment, **not learned or claimed to be optimal**.

Evaluate the hybrid on a selected, reproducible subset of MSH-WSD and compare it with the corresponding **Wu–Palmer-only baseline on exactly the same instances**. Report coverage and analyze both successes and failures. Since sound similarity does not necessarily imply semantic similarity, improvement is a hypothesis to test rather than a guaranteed result.

### 7. Interactive user interface

A planned **Streamlit** interface will allow users to:

- Enter a sentence or short biomedical passage and specify the ambiguous target word.
- Select one or more available WSD algorithms.
- View predicted senses, glosses or medical concept descriptions, and similarity scores.
- Compare algorithms side by side.
- Inspect a phonetic-distance matrix / heatmap.
- Explore the hybrid semantic and phonetic score components.
- View experiment summaries and per-category benchmark results.

### 8. Literature review and critical analysis

The final report will relate the implementation to the original Lesk method, WordNet semantic similarity, biomedical sense inventories, MSH-WSD benchmark design, and more recent concept-representation methods. It will critically examine sense-inventory mismatch, acronym ambiguity, corpus/domain shift, incomplete lexical coverage, dataset labeling assumptions, runtime, reproducibility, and the limitations of phonetic signals for semantic interpretation.

## Evaluation protocol

For each ambiguous entry `t`:

\[
Accuracy(t) = \frac{\#\text{ correct predictions for }t}{\#\text{ evaluated instances for }t}
\]

Primary headline results use the **macro-average** across eligible entries in each category:

\[
MacroAccuracy(T) = \frac{1}{|T|}\sum_{t\in T} Accuracy(t)
\]

We will also report **micro-accuracy** (correct predictions divided by evaluated instances across the category), prediction coverage, the number of excluded/unmapped instances, and per-entry accuracy. Explicitly defined policies will distinguish a full-population accuracy (where abstentions count as incorrect) from accuracy conditional on a prediction, so algorithms with low coverage are not unfairly favored.

For meaningful comparisons, all algorithms will run on the **same eligible examples**, with consistent preprocessing, candidate senses, gold-label alignment, random seed, and aggregation rules. Any use of information-content corpora, medical ontologies, or extra gloss resources will be recorded.

## Planned repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                      # Streamlit interface
├── data/
│   ├── README.md               # Instructions for obtaining MSH-WSD
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Tokenization, normalization, lemmatization
│   ├── dataset.py              # Dataset parsing, sampling and labels
│   ├── sense_alignment.py      # WordNet ↔ UMLS sense handling
│   ├── lesk_baseline.py        # NLTK Lesk baseline
│   ├── pywsd_methods.py        # Comparison algorithms
│   ├── phonetic.py             # Encoding, edit distance and matrices
│   ├── hybrid_wsd.py           # Wu–Palmer + phonetic scorer
│   └── evaluation.py           # Predictions, metrics and coverage
├── scripts/
│   ├── run_baseline.py
│   ├── run_sample_experiment.py
│   ├── run_full_benchmark.py
│   └── run_hybrid_experiment.py
├── tests/
├── results/                    # Tables and figures (generated)
└── docs/                       # Report, methodology and references
```

> This is the **target structure**, not a statement that these files have already been implemented. The tree and commands below will be updated as development proceeds.

## Installation (planned)

Use a supported Python 3 environment; dependency versions will be pinned after compatibility testing, particularly for older `pywsd` and `fuzzy` releases.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4 punkt punkt_tab wordnet_ic
```

Expected dependencies include `nltk`, `pywsd`, `fuzzy`, an edit-distance package, `numpy`, `pandas`, `matplotlib`, `seaborn` (optional), and `streamlit`. The final tested versions and any platform-specific installation steps will be documented in `requirements.txt`.

### Usage (after implementation)

```bash
python scripts/run_baseline.py
python scripts/run_sample_experiment.py --seed 42
python scripts/run_full_benchmark.py
python scripts/run_hybrid_experiment.py --seed 42
streamlit run app.py
```

These commands describe the **intended command-line interface**; they will become executable when the corresponding scripts have been created.

## Results

**No experimental accuracy values are reported yet.** The following summary will be populated from actual reproducible runs:

| Method | Regular terms: macro accuracy | Acronyms: macro accuracy | Coverage |
|---|---:|---:|---:|
| Standard Lesk | TBD | TBD | TBD |
| Adapted Lesk | TBD | TBD | TBD |
| Extended Lesk | TBD | TBD | TBD |
| Cosine Lesk | TBD | TBD | TBD |
| Path similarity | TBD | TBD | TBD |
| Information-content similarity | TBD | TBD | TBD |
| Highest lemma count | TBD | TBD | TBD |
| Wu–Palmer baseline | TBD | TBD | TBD |
| Wu–Palmer + phonetic (0.5/0.5) | TBD | TBD | TBD |

The final analysis will include per-term examples, error categories, runtime comparisons, and a focused ablation comparing the hybrid model with the semantic-only version.

## References

1. Lesk, M. (1986). *Automatic sense disambiguation using machine readable dictionaries: How to tell a pine cone from an ice cream cone*. Proceedings of SIGDOC '86. https://doi.org/10.1145/318723.318728
2. Banerjee, S., & Pedersen, T. (2002). *An adapted Lesk algorithm for word sense disambiguation using WordNet*. CICLing 2002. https://aclanthology.org/W02-1003/
3. Fellbaum, C. (Ed.). (1998). *WordNet: An Electronic Lexical Database*. MIT Press. https://wordnet.princeton.edu/
4. Wu, Z., & Palmer, M. (1994). *Verb semantics and lexical selection*. Proceedings of ACL 1994. https://aclanthology.org/P94-1019/
5. Resnik, P. (1995). *Using information content to evaluate semantic similarity in a taxonomy*. IJCAI 1995. https://arxiv.org/abs/cmp-lg/9511007
6. Jimeno-Yepes, A. J., McInnes, B. T., & Aronson, A. R. (2011). *Exploiting MeSH indexing in MEDLINE to generate a data set for word sense disambiguation*. BMC Bioinformatics, 12, 223. https://doi.org/10.1186/1471-2105-12-223
7. Tulkens, S., Šuster, S., & Daelemans, W. (2016). *Using Distributed Representations to Disambiguate Biomedical and Clinical Concepts*. BioNLP Workshop 2016. https://aclanthology.org/W16-2910/
8. NLTK Word Sense Disambiguation documentation: https://www.nltk.org/howto/wsd.html
9. `pywsd` repository: https://github.com/alvations/pywsd
10. Yarn repository: https://github.com/clips/yarn
11. MSH-WSD dataset card: https://huggingface.co/datasets/bigbio/msh_wsd
12. Streamlit documentation: https://docs.streamlit.io/

## Limitations and responsible use

This is an **academic NLP research project**, not a clinical decision-support or diagnostic tool. General-purpose WordNet senses may not correspond directly to biomedical/UMLS concepts. MSH-WSD labels derive from an automatic indexing-based construction method, and the dataset may not capture all real-world clinical ambiguity. Phonetic resemblance also does not imply shared meaning. Results must therefore be interpreted with reference to domain coverage, label alignment, and the evaluated population.

## License and attribution

A license for the **project's own code** will be selected before public release. External packages and data retain their respective licenses and access conditions. Cite the MSH-WSD paper and any third-party code or resources actually used in experiments.
