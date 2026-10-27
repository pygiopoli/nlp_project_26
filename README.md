# Word Sense Disambiguation in Medical Context

Knowledge-based Word Sense Disambiguation (WSD) experiments on biomedical text, evaluated on the
[MSH-WSD](https://lhncbc.nlm.nih.gov/ii/areas/WSD/collaboration.html) benchmark, plus a Django web app
that runs and compares WSD algorithms on a user-supplied sentence.

The assignment is in [project_31.md](project_31.md). Methods, results and discussion are in the
[technical report](https://www.overleaf.com/project/6ac8dfcb5043816f71cb841f/share#925442360e3c2fa9925fb68716f5c4f14e8b7d8c9e26725a).
This README covers only setup and usage.

## Contents

| Task | Description | Location |
|---|---|---|
| 1 | Standard and extended Lesk on the example sentence ("drug"), with and without POS restriction | [scripts/p1_2_lesk/drug_disamb.ipynb](scripts/p1_2_lesk/drug_disamb.ipynb) |
| 2 | NLTK Lesk on two selections of 10 acronyms + 10 terms of MSH-WSD, manual CUI → WordNet mapping, Lesk variants, MFS baseline and simplified Lesk | [scripts/p1_2_lesk/task_2.ipynb](scripts/p1_2_lesk/task_2.ipynb) |
| 3 | pywsd algorithms on example sentences | [scripts/p3_4_pywsd/task_3.ipynb](scripts/p3_4_pywsd/task_3.ipynb) |
| 4 | pywsd algorithms on MSH-WSD | [scripts/p3_4_pywsd/task_4.ipynb](scripts/p3_4_pywsd/task_4.ipynb) |
| 5 | Soundex + Levenshtein distance matrices, also on translations of the sentence | [scripts/p5_6_phonetic/p_distance.ipynb](scripts/p5_6_phonetic/p_distance.ipynb) |
| 6 | Wu–Palmer + phonetic similarity on MSH-WSD | [scripts/p5_6_phonetic/task_6.ipynb](scripts/p5_6_phonetic/task_6.ipynb) |
| 7 | Web GUI (Django) | [dis_project/](dis_project/) |
| 8 | Literature review | [technical report](https://www.overleaf.com/project/6ac8dfcb5043816f71cb841f/share#925442360e3c2fa9925fb68716f5c4f14e8b7d8c9e26725a) |
| 9 | Tasks 4 and 6 on the full dataset, with an automatic CUI → WordNet mapping validated against the manual one | [scripts/full_dataset/run_full_dataset.py](scripts/full_dataset/run_full_dataset.py) |
| 10 | Phonetic word matching (Soundex, edit distance) inside simplified Lesk | [scripts/p5_6_phonetic/phonetic_lesk.ipynb](scripts/p5_6_phonetic/phonetic_lesk.ipynb) |
| 11 | SapBERT embedding-based WSD over UMLS senses | [scripts/bert_extention/bert_wsd.ipynb](scripts/bert_extention/bert_wsd.ipynb) |

## Repository layout

```text
.
├── data/                       # the MSH-WSD corpus is not tracked into the repository beause of copyright
├── scripts/
│   ├── wsd_utils.py            # shared: MSH-WSD loader, acronym/term split, CUI → WordNet
│   │                           #   mapping, pywsd → NLTK sense translation, sampling, metrics
│   ├── p1_2_lesk/              # Tasks 1–2
│   ├── p3_4_pywsd/             # Tasks 3–4 (+ cached predictions)
│   ├── p5_6_phonetic/          # Tasks 5–6, utils.py (Soundex, Levenshtein)
│   ├── full_dataset/           # command-line pipeline for Tasks 4 and 6 on all words, results/
│   └── bert_extention/         # SapBERT extension (+ cached predictions)
├── dis_project/                # Django web app (Task 7)
│   ├── config/                 # settings, root URLconf, WSGI/ASGI
│   ├── algorithms/             # WSD algorithms, independent of Django (+ tests)
│   ├── wsd/                    # Django app: form, views, services, templates, static JS (+ tests)
│   ├── manage.py
│   └── requirements.txt        # GUI dependencies only (Django, NLTK, torch, transformers)
├── requirements.txt            # everything (includes dis_project/requirements.txt)
└── project_31.md               # assignment
```

## Prerequisites

- **Python 3.11 or newer.** The pinned pandas and numpy versions require 3.11. The code was developed on
  Python 3.11 and 3.13.
- **Internet access on the first run.** The code downloads the NLTK data and Open English WordNet 2024.
  The extension also downloads the SapBERT model, and the GUI loads Bootstrap and Chart.js from a CDN.
- **A UMLS Terminology Services (UTS) account**, needed to download MSH-WSD and to get a UMLS API key.
  The account is free, but NLM must approve the license request, which can take a few days. Tasks 1, 3
  and 5 and the GUI work without it, except the GUI's SapBERT algorithm.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/pygiopoli/nlp_project_26.git
cd nlp_project_26
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

With conda, use `conda create -n nlp python=3.11` and then `conda activate nlp`.

### 3. Install the dependencies

```bash
pip install -r requirements.txt     # notebooks, full-dataset pipeline, SapBERT extension and web GUI
```

To run only the GUI, `pip install -r dis_project/requirements.txt` is enough. Both files include torch and
transformers, which take about 2 GB. The versions are pinned to the ones the code was tested with.

`pywsd` 1.3.0 uses the [`wn`](https://github.com/goodmami/wn) package with **Open English WordNet 2024**,
while NLTK uses **WordNet 3.0**. Both are needed: the gold-standard mapping uses NLTK synset names, and
`wsd_utils.to_nltk` converts pywsd answers to those names through WordNet sense keys. Other pywsd
releases have not been tested.

### 4. Download the lexical resources

```bash
python -m nltk.downloader wordnet omw-1.4 stopwords punkt punkt_tab averaged_perceptron_tagger_eng
python -c "import wn; wn.download('oewn:2024')"
```

| Resource | Default location | Override |
|---|---|---|
| NLTK data | `~/nltk_data` (Linux/macOS), `%APPDATA%\nltk_data` (Windows) | `NLTK_DATA` |
| Open English WordNet 2024 | `~/.wn_data` | `WN_DATA_DIR` |

pywsd includes the information-content files for the Resnik, Jiang–Conrath and Lin measures, so you
don't need to download them.

### 5. Get the MSH-WSD corpus

1. Request a UMLS license and create a UTS account at <https://uts.nlm.nih.gov/uts/>.
2. Download the MSH WSD data set from the
   [NLM WSD page](https://lhncbc.nlm.nih.gov/ii/areas/WSD/collaboration.html).
3. Extract the archive so that the files end up in `data/MSHCorpus/` at the repository root:

```text
data/
├── MSHCorpus/
│   ├── benchmark_mesh.txt            # word → candidate CUIs
│   ├── README.txt
│   └── <word>_pmids_tagged.arff      # one file per ambiguous word (203 files)
└── umls_cui_info.json                # generated later by `run_full_dataset.py umls`
```

`scripts/wsd_utils.py` finds the corpus relative to its own location (`DATA_DIR`), so the path must match
this layout exactly. `data/` is in `.gitignore`: the corpus is licensed under UMLS terms and must not be
committed.

### 6. Get a UMLS API key (optional)

After signing in to UTS, copy the API key from your profile page. The key is entered at runtime with
`getpass` and is never written to disk. Two places use it:

- `task_2.ipynb` uses it only to show the UMLS name of each CUI. Without a valid key the names are
  empty, but the evaluation still runs.
- `run_full_dataset.py umls` uses it to download names and definitions for all CUIs. Both the full-dataset
  pipeline and the SapBERT extension need this step.

## Running the notebooks

```bash
jupyter lab
```

- **Run each notebook from its own folder.** This is the default in Jupyter and in VS Code. The notebooks
  import `wsd_utils` with `sys.path.append("..")`, and they read and write their caches with relative
  paths.
- Run the cells from top to bottom.

| Notebook | Needs MSH-WSD | Needs UMLS | Cache file |
|---|:---:|:---:|---|
| `p1_2_lesk/drug_disamb.ipynb` | | | |
| `p1_2_lesk/task_2.ipynb` | ✓ | API key (optional) | |
| `p3_4_pywsd/task_3.ipynb` | | | |
| `p3_4_pywsd/task_4.ipynb` | ✓ | | `task4_predictions.json` |
| `p5_6_phonetic/p_distance.ipynb` | | | |
| `p5_6_phonetic/task_6.ipynb` | ✓ | | `task6_predictions.json` |
| `bert_extention/bert_wsd.ipynb` | ✓ | `data/umls_cui_info.json` | `bert_predictions.json` |

**Caches.** The pywsd similarity methods and SapBERT are slow, so the evaluation loops save their predictions
to a JSON file next to the notebook after every word. Words already in the file are skipped. The repository
includes these files, so the result tables come straight from the saved predictions. Delete a cache file to
recompute it from scratch. `task_6.ipynb` also reads `../p3_4_pywsd/task4_predictions.json`, which is its
Wu–Palmer baseline.

**Shared parameters** in `scripts/wsd_utils.py`:

| Name | Value | Meaning |
|---|---|---|
| `WINDOW` | 10 | context words on each side of the target (Tasks 4, 6 and the extension) |
| `N_PER_SENSE` | 10 | abstracts sampled for each sense (sampling seed 0) |
| `MAPPING` | 39 words | manual CUI → WordNet mapping (both Task 2 selections, seeds 42 and 7) |

## Full-dataset pipeline (Tasks 4 and 6 on all words)

Run it from its folder, because it uses relative paths:

```bash
cd scripts/full_dataset

python run_full_dataset.py info          # number of words and CUIs (quick check)
python run_full_dataset.py umls          # prompts for the UMLS API key → data/umls_cui_info.json
python run_full_dataset.py mapping [N]   # automatic CUI → WordNet mapping → results/auto_mapping.json
                                         #   N = minimum word overlap (default 4)
python run_full_dataset.py task4         # Task 4 methods → results/task4_predictions.json
python run_full_dataset.py task6         # Wu–Palmer + phonetic → results/task6_predictions.json
python run_full_dataset.py results       # prints the tables → results/task4_results.csv, task6_results.csv
```

- Each step reads the output of the previous ones.
- `umls` saves every 20 CUIs, and `task4` and `task6` save after every word. You can interrupt them and
  run them again to resume.
- `results` also works on partial predictions.
- The repository already includes `results/` (mapping and predictions), so `results` works without running
  the earlier steps. `data/umls_cui_info.json` is not tracked, though: run `umls` locally if you need to
  rebuild the mapping or use the SapBERT extension.

## SapBERT extension

Before running `scripts/bert_extention/bert_wsd.ipynb`:

1. Install `requirements.txt` (step 3 of Setup). The first cell of the notebook runs `%pip install` for the
   same packages; skip it if they are already installed.
2. Run `python run_full_dataset.py umls` to create `data/umls_cui_info.json`. The notebook builds each
   sense description from the names and definitions in this file.

On the first run, the notebook downloads the model `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
(about 0.4 GB) to the Hugging Face cache (`~/.cache/huggingface`, override with `HF_HOME`). The code
does not move the model to a GPU, so it runs on CPU.

## Web GUI (Task 7)

The GUI is a Django 5.2 app with no database. The user enters a sentence and a target word, chooses one
algorithm or all of them, and sees, for each algorithm:

- the chosen sense;
- the full ranking of candidate senses, with a score chart;
- the overlapping words highlighted in the sentence and in the definition;
- the POS tags;
- whether the algorithms agree.

### Run

The project root environment works. A separate environment that has only `dis_project/requirements.txt`
also works.

```bash
cd dis_project
pip install -r requirements.txt     # Django, NLTK, torch, transformers (skip if already installed)
python -m algorithms.nltk_data      # NLTK packages used by the app
python manage.py runserver
```

Then open <http://127.0.0.1:8000/>.

- You don't need to run `migrate`, because `DATABASES` is empty and no app uses models.
- When `runserver` starts, WordNet and the POS tagger load in the serving process (`WsdConfig.ready`).
  Startup takes a few seconds, but the first request is fast.
- The form uses GET, so each result has a shareable URL, for example
  `/?sentence=The+patient+has+a+cold&target=cold&algorithm=all`.

| `algorithm` | Algorithm |
|---|---|
| `lesk` | Standard Lesk (same scoring as `nltk.wsd.lesk`, with the full ranking) |
| `lesk_pos` | Standard Lesk, candidates limited to the POS of the target |
| `extended_lesk` | Extended Lesk: definition, examples and related synsets |
| `extended_lesk_pos` | Extended Lesk, candidates limited to the POS of the target |
| `wu_palmer` | Wu–Palmer similarity with the senses of the context words (Task 3) |
| `wu_palmer_phonetic` | 0.5 · Wu–Palmer + 0.5 · phonetic similarity (Task 6) |
| `sapbert` | SapBERT embeddings, choosing among the UMLS senses (CUIs) of MSH-WSD (extension) |
| `all` | Runs every registered algorithm and compares them |

The WordNet-based algorithms work with any word. `sapbert` instead works only on the ambiguous words of
MSH-WSD (e.g. *cold*, *ventricle*), and gives no candidates for other words. It needs two files that are
not in the repository: `data/MSHCorpus/benchmark_mesh.txt` (step 5 of Setup) and `data/umls_cui_info.json`
(`run_full_dataset.py umls`). The first time an MSH-WSD word is disambiguated, the app downloads and loads
the SapBERT model, which takes a while. Because its senses are UMLS concepts, its choice never matches the
WordNet senses of the other algorithms in the comparison.

### Configuration

| Environment variable | Default | Effect |
|---|---|---|
| `DJANGO_SECRET_KEY` | development-only key | Secret key |
| `DJANGO_DEBUG` | `1` | `0`, `false` or `no` turns debug off. Without debug, `runserver` stops serving static files, so start it with `--insecure` to keep the charts. |

`ALLOWED_HOSTS` accepts only `127.0.0.1` and `localhost`. The app is meant for local use with `runserver`
and is not set up for deployment.

### Tests

```bash
cd dis_project
python manage.py test               # every test (algorithms + wsd)
python manage.py test algorithms    # only the algorithm tests
```

The tests need the NLTK data from `python -m algorithms.nltk_data`. They don't need the MSH-WSD data, but the
first run downloads the SapBERT model.

### Adding an algorithm

1. In `algorithms/`, write a function `(sentence: str, target: str) -> WSDResult` (see
   `algorithms/base.py`). The `algorithms` package must not import Django.
2. Add an `AlgorithmSpec` for it to `REGISTERED_SPECS` in `algorithms/registry.py`.

The form, the comparison mode and the result page find new algorithms through the registry, so you don't
need to change anything else.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `LookupError: Resource ... not found` from NLTK | Run the `nltk.downloader` command from step 4 (or `python -m algorithms.nltk_data` for the GUI). |
| Error about `oewn:2024` when importing `wsd_utils` or `pywsd` | Run `python -c "import wn; wn.download('oewn:2024')"`. |
| `FileNotFoundError` for `data/MSHCorpus/benchmark_mesh.txt` | The corpus is missing or in the wrong place (see step 5). |
| `ModuleNotFoundError: No module named 'wsd_utils'` | The notebook or script is not running from its own folder. |
| `FileNotFoundError` for `data/umls_cui_info.json` (notebook, or SapBERT card in the GUI) | Run `python run_full_dataset.py umls` from `scripts/full_dataset`. |
| SapBERT card shows `FileNotFoundError` for `benchmark_mesh.txt` | The MSH-WSD corpus is missing (see step 5). |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope Process RemoteSigned`, then activate again. |
| GUI page without styling or charts | The browser cannot reach the CDN, or `DJANGO_DEBUG=0` is set and `runserver` was started without `--insecure`. |

## Data and licensing

MSH-WSD and the UMLS content it references are covered by the UMLS license. Each user must download
them under their own license. Do not redistribute them or commit them to this repository.

If you use the corpus, cite:

> Jimeno-Yepes, A. J., McInnes, B. T., & Aronson, A. R. (2011). *Exploiting MeSH indexing in MEDLINE to
> generate a data set for word sense disambiguation.* BMC Bioinformatics, 12, 223.
> https://doi.org/10.1186/1471-2105-12-223

The project is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license. Third-party packages keep their own licenses.

## Authors

- Giorgio Alleva
- Francesca Lonardo
