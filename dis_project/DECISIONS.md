# Decisions – Django WSD visualizer (Project 31, task 7)

Summary of the decisions taken before writing code. Update this file whenever
a decision changes. Deadline: **31 October 2026**.

## Environment

| Topic | Decision |
|---|---|
| Location | Django project in `nlp_project_26/dis_project/` (`manage.py`, `config/`, `wsd/`) |
| Python | 3.11 (system Python 3.11.4) |
| Environment | `venv` + `pip`, virtual environment in `dis_project/.venv` |
| Django | 5.2 LTS (Django 6.x requires Python >= 3.12) |
| Dependencies | `django` and `nltk` only, pinned in `requirements.txt` |
| Git | Existing repository `nlp_project_26`; one commit per step, run by the user |

## Conventions

| Topic | Decision |
|---|---|
| Language | Everything in English (code, UI, docs, commit messages) |
| Tests | Django test runner (`python manage.py test`). Because there is no database, Django tests use `SimpleTestCase` (`TestCase` needs a database); algorithm tests may use plain `unittest.TestCase` |
| Formatter / linter / type checker | None for now |

## Structure

| Topic | Decision |
|---|---|
| Project / app names | `config` / `wsd` |
| Settings | Single `settings.py`; `SECRET_KEY` and `DEBUG` read from environment variables (`DJANGO_SECRET_KEY`, `DJANGO_DEBUG`) with development defaults |
| Database / API | None: `DATABASES = {}`; the database-backed apps (`admin`, `auth`, `contenttypes`, `sessions`) and `messages` are removed, with their middleware, context processors and the `admin/` URL |
| Scope | Local only (`runserver`), never deployed |

## Algorithms

| Topic | Decision |
|---|---|
| Existing code to port | `scripts/p2/drug_disamb.ipynb`: standard Lesk (`nltk.wsd.lesk`) and extended Lesk (`extended_lesk` and helpers), each with or without the POS filter |
| Future algorithms | Potentially many: the pywsd family (which uses OEWN ids, not NLTK synset names), the Wu-Palmer + phonetic hybrid (task 6), embedding / BERT models. Senses must therefore be identified by plain strings, never by NLTK objects |
| URLs | Possibly one URL per algorithm (to be discussed in steps 3 and 7) |
| Per-algorithm options, comparison mode, resource caching | To be discussed in steps 3-4 |

### Adaptations of the notebook code (step 2)

| Topic | Decision |
|---|---|
| Standard Lesk ranking | Reimplement the `nltk.wsd.lesk` scoring (overlap between `set(tokens)` and `definition().split()`, ties won by the first synset in WordNet order) to expose every candidate's score and overlap. A test checks that the winner matches `nltk.wsd.lesk` |
| Common call signature | Both variants are called as `(sentence, target, use_pos)`; tokenization, tagging and target lookup happen inside. The bodies of `related_synsets`, `extended_gloss` and `context_lemmas` stay unchanged |
| Target lookup | Match `lemma(token)` against `lemma(target)`, both lowercased, so `drugs`, `Drug` and `HIV` are found |
| No synsets | Return an empty result ("no candidates") instead of raising (`max()` on an empty sequence in the notebook) |
| Tie / no evidence | Kept as metadata, so the UI can say the choice is a fallback to the most frequent sense |
| Behavioral quirks | Kept on purpose, to match the notebook and the report: standard Lesk is case-sensitive, keeps stopwords and the target word; extended Lesk lemmatizes glosses as nouns only |
| NLTK data | Not downloaded at import time; a separate setup step (discussed in step 4) |
| Notebook | Left untouched; the Django module is an adapted copy (intentional duplication) |
| Target missing from the sentence | Rejected by form validation, for every algorithm |

### Algorithm interface (step 3)

| Topic | Decision |
|---|---|
| Location | Top-level package `dis_project/algorithms/`, sibling of the `wsd` app, with no Django imports |
| Result structure | Frozen dataclasses in `algorithms/base.py`: `Sense`, `ScoredSense`, `WSDResult` (plain strings and numbers, no NLTK objects); `score` is a `float` |
| Evidence | Two generic optional fields on `ScoredSense`: `matched_words` (highlighted in the UI) and `details` (key/value table) |
| Conversion | Each algorithm converts its own synsets into `Sense` objects |
| Registration | Plain function + `AlgorithmSpec` (key, label, description, `run(sentence, target)`) added to `REGISTERED_SPECS` in `algorithms/registry.py`; `build_registry` rejects duplicate keys. A `Protocol` or class is introduced only when a stateful algorithm (e.g. a loaded model) needs it |
| POS filter | Four registered variants: `lesk`, `lesk_pos`, `extended_lesk`, `extended_lesk_pos` (two functions with a `use_pos` parameter); no per-algorithm options in the form for now |
| URLs / comparison | Tentatively one page with a multi-select (`?algorithm=a&algorithm=b`) supporting any subset; to be decided in steps 6-7 |

### Resources and porting (step 4)

| Topic | Decision |
|---|---|
| Resource loading | Lazy: NLTK loads WordNet (~2.5 s) and the tagger once per process on first use; our derived resources (stopword set, lemmatizer) use `functools.cache`. No warm-up in `AppConfig.ready()` |
| NLTK data download | `python -m algorithms.nltk_data` (list in `REQUIRED_NLTK_DATA`): `wordnet`, `stopwords`, `punkt_tab`, `averaged_perceptron_tagger_eng` |
| NLTK data location | NLTK's default user folder (shared with the notebooks) |
| Fallback flag | Same rule for every Lesk variant: `is_fallback` when the best score is 0 or shared by several candidates |
| Shared helpers | `algorithms/nltk_utils.py`: `find_target` (also used by the form), `sense_from_synset`, `rank` |

## Look and feel

| Topic | Decision |
|---|---|
| Theme | Light only |
| Bootstrap 5.3 / Chart.js | **Loaded from a CDN.** This changes the original "no CDN, works offline" requirement, by explicit choice. Exact versions are pinned and loaded with SRI `integrity` hashes; the demo needs an internet connection |
| Chart and evidence style | To be discussed in steps 9-10 |

## Cleanup

| Topic | Decision |
|---|---|
| Streamlit prototype | `scripts/p7/app.py` and `scripts/p7/lesk.py` will be deleted and README section 7 updated, in a dedicated later step |
