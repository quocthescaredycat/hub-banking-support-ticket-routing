# BANKING77: intent classification with classical ML

Course project code for the plan in [`PLAN.md`](PLAN.md): fine-grained banking intent classification with
lexical TF-IDF and frozen sentence-embedding representations, Logistic Regression and linear SVM classifiers,
and a development protocol that keeps the official test set locked until the configuration is frozen.

Current development results: [`report/dev_results_lexical.md`](report/dev_results_lexical.md).

## Layout

```text
banking_nlp/
  data.py           pinned source + checksums, label mapping, duplicate audit, frozen partitions, test lock
  preprocessing.py  lexical variants L0-L6 (normalization, contractions/negation, lemmas, numbers, stop words)
                    and light normalization for the sentence encoder
  models.py         TF-IDF (word / char_wb / word+char) and MiniLM classifiers (LR, LinearSVC)
  evaluation.py     macro metrics over 77 labels, top-3, bootstrap CIs, paired deltas, confusions, latency
  experiments.py    stages: prepare / lexical / embeddings / select / final (gated)
  figures.py        report figures
  artifacts.py      JSON, hashes, environment capture
configs/experiment.json   pre-registered grid and selection rules
notebooks/banking77_experiments.ipynb   controller notebook (calls the modules)
artifacts/          manifests, grids, candidates, validation predictions (models/embeddings are git-ignored)
report/             results write-up and figures
tests/              contract tests (no full training needed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the stages

```bash
python -m banking_nlp.experiments prepare      # verify source, audit, load/freeze partitions
python -m banking_nlp.experiments lexical      # C1/C2 + preprocessing ablation (~15 min on 2 CPUs)
python -m banking_nlp.experiments embeddings   # C3/C4: needs sentence-transformers + Hugging Face access
python -m banking_nlp.experiments select       # latency on this machine + deployment choice -> artifacts/frozen/
python -m banking_nlp.experiments final --i-confirm-freeze   # once, after the freeze
python -m pytest -q
```

`prepare` downloads the three source files from the PolyAI repository at commit `57ec275` and refuses to
continue if any checksum differs. `splits.csv` is the frozen partition assignment; it is loaded, never
recomputed, once it exists.

The model files (`artifacts/models/*.joblib`, ~20 MB each) are not in git. If they are missing, re-run the
`lexical` stage; it rewrites the candidate records with the new file hashes. `select` checks those hashes.

### Running C3/C4

The lexical results were produced in an environment that could not reach huggingface.co, so the MiniLM stage
has only been tested with a stand-in encoder (`tests/test_embedding_stage.py`). On a machine with access:

1. `python -m banking_nlp.experiments embeddings` (CPU is fine for ~10k short requests).
2. Copy `encoder.resolved_revision` from `artifacts/embeddings/minilm_dev_meta.json` into
   `configs/experiment.json` → `embedding.revision`, and pin the `sentence-transformers` version that worked in
   `requirements.txt`.
3. Run `select`.

## Protocol in one paragraph

Official training is split 80/10/10 into fitting, model-validation, and routing-validation partitions with
duplicate-group-aware stratification (seed 42). TF-IDF vocabularies and classifiers are fitted on the fitting
partition only; preprocessing is stateless. Model and preprocessing choices use model-validation macro-F1 under
the rules in `configs/experiment.json`. The routing partition is reserved for the review threshold. Test
loaders raise `TestSetLockedError` until `selection.json` exists and the caller confirms the freeze; all four
configurations are then scored once from persisted predictions.

## Data and licence

BANKING77 (Casanueva et al., 2020), CC-BY-4.0, from
[PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets). The raw files are
downloaded at run time and not committed; `artifacts/data/splits.csv` contains the training texts with
attribution as required by the licence.
