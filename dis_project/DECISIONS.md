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
