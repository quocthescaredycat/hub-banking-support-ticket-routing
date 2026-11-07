# BANKING77: intent classification with classical ML

Course project code for the plan in [`PLAN.md`](PLAN.md): fine-grained banking intent classification with
lexical TF-IDF and frozen sentence-embedding representations, Logistic Regression and linear SVM classifiers,
a review threshold for routing, and a development protocol that keeps the official test set locked until the
configuration is frozen.

Results:

- [`report/results_full.md`](report/results_full.md): C1–C4 on model validation, the frozen deployment choice,
  the routing threshold, and the single official test pass (with bootstrap intervals, paired differences,
  overlap sensitivity and the lexical-difficulty slice).
- [`report/dev_results_lexical.md`](report/dev_results_lexical.md): data audit and the preprocessing ablation.

## Layout

```text
banking_nlp/
  data.py           pinned source + checksums, label mapping, duplicate audit, frozen partitions, test lock
  preprocessing.py  lexical variants L0-L6 (normalization, contractions/negation, lemmas, numbers, stop words)
                    and light normalization for the sentence encoder
  models.py         TF-IDF (word / char_wb / word+char) and MiniLM classifiers (LR, LinearSVC)
  evaluation.py     macro metrics over 77 labels, top-3, bootstrap CIs, paired deltas, confusions, latency
  routing.py        review threshold on routing_val; lexical-difficulty test slice
  experiments.py    stages: prepare / lexical / embeddings / select / routing / final (gated)
  figures.py        report figures
  artifacts.py      JSON, hashes, environment capture
configs/experiment.json   pre-registered grid, selection rules, pinned MiniLM revision
notebooks/banking77_experiments.ipynb   controller notebook (calls the modules)
artifacts/          manifests, grids, candidates, frozen choices, predictions, final metrics
                    (models/ and embeddings/ are git-ignored)
report/             results write-ups and figures
tests/              contract tests (no full training needed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions that produced the frozen artifacts; see its comments for the torch build.

## Run the stages

```bash
python -m banking_nlp.experiments prepare      # verify source, audit, load/freeze partitions
python -m banking_nlp.experiments lexical      # C1/C2 + preprocessing ablation (~15 min on 2 CPUs)
python -m banking_nlp.experiments embeddings   # C3/C4: frozen MiniLM at the pinned revision
python -m banking_nlp.experiments select       # latency on this machine + deployment choice -> artifacts/frozen/
python -m banking_nlp.experiments routing      # review threshold on routing_val -> artifacts/frozen/routing.json
python -m banking_nlp.experiments final --i-confirm-freeze   # once: test predictions, then scoring
python -m pytest -q
```

`prepare` downloads the three source files from the PolyAI repository at commit `57ec275` and refuses to
continue if any checksum differs. `splits.csv` is the frozen partition assignment; it is loaded, never
recomputed, once it exists.

`final` refuses to read any test input unless `selection.json` exists, `routing.json` exists and was made from
that exact `selection.json`, and `--i-confirm-freeze` is given. It predicts all four families in memory and writes
files only if every family succeeded; it refuses to overwrite existing test predictions. Scoring then reads only
the persisted predictions. **The final pass for this project has already been run (7 Nov 2026)**; re-scoring
from the saved predictions is `banking_nlp.experiments.score_test(".", i_confirm_configuration_is_frozen=True)`.

The model files (`artifacts/models/*.joblib`) and the MiniLM embedding cache are not in git. If they are missing,
re-run the `lexical` and `embeddings` stages; the candidate records then get new file hashes, so a new
`select`/`routing` freeze would be a new experiment, not the recorded one.

## Protocol in one paragraph

Official training is split 80/10/10 into fitting, model-validation, and routing-validation partitions with
duplicate-group-aware stratification (seed 42). TF-IDF vocabularies and classifiers are fitted on the fitting
partition only; preprocessing is stateless. Model and preprocessing choices use model-validation macro-F1 under
the rules in `configs/experiment.json`. The routing partition is used only to freeze the review threshold of the
selected model. Test loaders raise `TestSetLockedError` until both freezes exist and the caller confirms them; all
four configurations are then scored once from persisted predictions.

## Data and licence

BANKING77 (Casanueva et al., 2020), CC-BY-4.0, from
[PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets). The raw files are
downloaded at run time and not committed; `artifacts/data/splits.csv` contains the training texts with
attribution as required by the licence.
