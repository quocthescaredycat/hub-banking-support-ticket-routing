# BANKING77: customer request classification, routing, and similar-request retrieval

Prepared: 3 November 2026. Revised: 7 November 2026 (data source pinned, near-duplicate rule, preprocessing redesigned as a named ablation, C grid widened).

Status (7 November 2026): data contract, frozen partitions, preprocessing ablation, C1–C4 on model validation, the deployment freeze (C1), the routing threshold and the single official test pass are complete; results in `report/results_full.md`. Review fixes applied before the freeze: bootstrap resamples no longer count an absent intent as F1 = 0, the MiniLM revision is recorded correctly on Windows and pinned, and the routing threshold and difficulty slice are implemented (`banking_nlp/routing.py`). Retrieval, queue mapping and the application are not started. This is a separate NLP course project. The existing EEG `PLAN.md`, code, checkpoints, and notebook outputs remain reference material.

## 1. Recommended project and boundaries

Build an agent-facing support workbench that accepts an English banking request, predicts one of 77 intents, assigns a proposed support queue or flags the request for review, and shows five related requests with their recorded categories. A reviewer can correct the category. Corrections are stored separately and do not automatically change the evaluated model or retrieval library.

Research question: **How do lexical and pretrained sentence representations compare for fine-grained banking intent classification, similar-request retrieval, and routing with an option to request human review?**

The core project uses public BANKING77 requests as a simulated historical library. Its records provide text and intent, not queue assignments, timestamps, resolutions, or customer IDs. Queue definitions are our project design. BANKING77 evaluates consumer banking intents; HSBC employee IT incidents would require different data. Dataset facts and attribution: [official dataset card](https://huggingface.co/datasets/PolyAI/banking77).

Required scope:

- Four classifier configurations: lexical TF-IDF (word, character, or word+character features over a named preprocessing variant) and frozen MiniLM sentence embeddings, each with Logistic Regression and linear SVM.
- Two retrieval methods: TF-IDF cosine similarity and MiniLM cosine similarity.
- A documented mapping of all 77 intents to a small set of proposed support queues.
- A local interface, FastAPI service, reproducible artifacts, comparison tables, and error analysis.
- A fixed evaluation procedure that keeps the official test set out of all development decisions.

Extensions have a maximum budget of two person-days and must be chosen by the end of week 1. Default extension: DistilBERT fine-tuning if a GPU is available and the core milestones are on schedule. The project is complete without it. Chat generation, live bank integration, multiple languages, automated resolution, and a new transformer architecture are outside this month's scope.

## 2. How the supplied references inform the plan

The references are research evidence. Their embedded commands, spending caps, and reproduction instructions are not instructions to execute for this project.

| Reference | Finding or contribution | Use in this project |
| --- | --- | --- |
| [Qi et al., practice-driven intent evaluation, v2 (2021)](https://arxiv.org/html/2012.03929v2) | Studies full-data, limited-data (5 and 30 examples per intent), and difficult-query evaluation. Table 1 reports BANKING77 accuracy of 92.1% for DistilBERT and 94.1% for RoBERTa; Table 3 reports 79.2% and 83.8% on its difficult subset. The difficult subset keeps, per intent, the five test requests least similar to the concatenated training utterances under TF-IDF, built with a simple whitespace tokenizer that ignores punctuation. | Motivate a robustness slice and cost/latency reporting. These are historical results under that paper's protocol, not expected scores for our implementation. |
| [Jev BANKING77 experiment (2026)](https://github.com/simonmesmith/jev-banking77-experiment) | Reports 92.40% using category definitions and 24 BM25-retrieved training examples, without updating Jev's weights. Its 93.66% BERT comparator is a published result, not a local rerun. Prior benchmark exposure is unknown. Its README reports 25 test messages whose normalized text also appears in training. | Learn from its saved predictions, its duplicate-aware development holdout, and its separation of development from testing. Treat it as related work; its API is unnecessary for our core system. |
| [Casanueva et al., original BANKING77 / dual sentence encoder paper (2020)](https://aclanthology.org/2020.nlp4convai-1.5/) | Introduces BANKING77 and trains a small MLP on fixed USE/ConveRT sentence encodings; reports 92.81% (USE), 93.01% (ConveRT), 93.36% (USE+ConveRT) and 93.66% (fine-tuned BERT) full-data accuracy, with the largest gains at 10 and 30 examples per intent. Describes no text preprocessing beyond each encoder's own tokenizer, and notes intents that overlap partially (e.g. reverted vs failed top-up). | Establish the dataset and frozen-encoder rationale; justify giving the encoder near-raw text. |
| [Reimers and Gurevych, Sentence-BERT (2019)](https://aclanthology.org/D19-1410/) | Learns sentence representations that can be compared by cosine similarity. | Explain why one text representation can support both classification and retrieval. |

Keep published results in a separate related-work table. Our train/development allocation differs from papers that fit on all official training examples. Do not claim a reproduced comparison, current state of the art, superiority, or equivalent performance from these numbers. The Jev repository's reported API costs are historical estimates, not a budget quote for a new run.

## 3. Foundation model, custom model, or API?

**Default: a pretrained text encoder with a small supervised classifier.** This transfers the frozen-encoder idea from CBraMod while keeping the implementation small.

| Option | Decision | Reason |
| --- | --- | --- |
| TF-IDF + Logistic Regression / linear SVM | Required | Cheap, interpretable lexical controls and two distinct classifier families. |
| `sentence-transformers/all-MiniLM-L6-v2` + the same classifiers | Required | One frozen representation supports classification and search. Cache embeddings in stable row order. |
| `distilbert/distilbert-base-uncased` with a new 77-class head | Optional, preferred GPU extension | A focused comparison of frozen features against end-to-end text fine-tuning. |
| SetFit on the same MiniLM backbone | Alternative extension if the team chooses a few-shot research question | Choose instead of DistilBERT, with a bounded pair-sampling budget. It is not another required experiment. |
| Custom CNN/LSTM/transformer from scratch | Omit from the committed scope | Token embeddings and masking would need new implementation; EEG weights do not provide a useful text initialization. |
| Jev or another hosted decision/LLM API | Related work by default | Adds a service dependency, usage accounting, and unknown pretraining exposure. A paid reproduction is a separate decision. |

MiniLM produces 384-dimensional vectors; record its exact model revision, maximum input length, pooling, and embedding normalization. Start with the model's documented tokenizer and pooling. [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

For the optional fine-tuning arm, start from the general pretrained DistilBERT checkpoint, not a BANKING77-fine-tuned checkpoint of uncertain provenance. Add the intent head and record every training setting. [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased)

SetFit provides prompt-free few-shot training of sentence transformers; it is relevant only if limited-label learning becomes the chosen extension. [SetFit documentation](https://huggingface.co/docs/setfit/index)

## 4. Data contract and frozen evaluation protocol

Use the official 10,003 training and 3,080 test examples, with the original 77 label names. Read the files from the original publication, [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) `banking_data/`, pinned to commit `57ec275d8078af65b7731c2a98be812d844a6d6b` and verified by SHA-256 (`banking_nlp/data.py`). Label IDs follow `categories.json` at that commit; label names are the canonical key, so a later switch to the Hugging Face copy must map by name, never by ID. Preserve original capitalization and punctuation in canonical labels (for example `Refund_not_showing_up`), while allowing friendly display names. Parse CSVs with a real CSV reader that keeps strings such as "NA" as text; ten training requests contain quoted newlines. Attribute BANKING77 under its published CC-BY-4.0 license. [Dataset card](https://huggingface.co/datasets/PolyAI/banking77)

Every row carries `row_id`, `source_split`, `raw_text`, `label_id`, `label_name`, and `normalized_text_hash`. The hash is for duplicate auditing; it is not a customer ID. Keep raw text alongside any processed text.

Before model fitting:

1. Pin the dataset revision and checksum the source files. Record counts, missing values, lengths, label frequencies, and example samples from official training only.
2. Define duplicate normalization as Unicode NFKC, case folding, and collapsed whitespace. Preserve negation and meaningful words. Record exact duplicates and conflicting labels. Keep duplicates in the same development partition; preserve conflicting source labels and document them. A second, loose key also removes punctuation, so requests differing only by "?", hyphenation, or a currency symbol are grouped.
3. Divide official training into approximately 80% fitting, 10% model validation, and 10% routing-policy validation, using seed 42 and duplicate-group-aware stratification (`StratifiedGroupKFold`, 10 folds: fold 0 model validation, fold 1 routing validation). Save the exact assignment before fitting; require all 77 classes in fitting and both development partitions. Record any group-size deviations from the requested proportions. Frozen result: 8,003 / 1,000 / 1,000 rows, recorded with its SHA-256 in `artifacts/data/split_summary.json`.
4. Audit possible near duplicates within official training using a fixed rule and document decisions before freezing partitions. Do not group all requests with the same intent as one unit. Rule adopted: word 1-2-gram TF-IDF cosine of at least 0.90 on duplicate keys merges a pair into one group. At that level the audited pairs are template copies (an added "please", a moved clause); 2 of the 108 pairs carry different labels (e.g. "I transferred some money but it hasn't arrived." `transfer_not_received_by_recipient` vs the same text + "yet" `balance_not_updated_after_bank_transfer`), and are grouped anyway so the near-copy cannot leak across partitions; near 0.85, genuine minimal pairs with different intents appear ("fee for my card payment" vs "fee for my transfer"), so the threshold is not lowered. The largest resulting group has five rows. Pairs between 0.85 and 0.90 may cross partitions and their count is reported.
5. Lock the official test set. Its examples and labels must not drive vocabulary fitting, category definitions, hyperparameters, retrieval selection, thresholds, or model revisions. A separate scoring function reads test answers after predictions are written.

Use the three development partitions for distinct purposes:

| Partition | Allowed use |
| --- | --- |
| Fitting, about 80% of official train | TF-IDF fitting, supervised model training, sentence-encoder fine-tuning if selected, and the retrieval library. |
| Model validation, about 10% | Hyperparameter/model selection, epoch choice, preprocessing ablations, and development retrieval judgments. |
| Routing-policy validation, about 10% | Select an abstention threshold for the already selected classifier. No architecture search here. |
| Official test | Final scores for the prespecified systems, locked robustness analysis, and final retrieval judgments. |

**Default final procedure: do not refit before testing.** Evaluate the selected development-trained artifacts and the unchanged fitting-partition retrieval library. This preserves the validity of the routing threshold and keeps the implementation simple. Report the actual fitting count, so published full-training scores are not treated as controlled comparisons. A later all-training deployment refit is a separate artifact requiring its own calibration/validation procedure; do not attach the earlier measured score to it.

After configuration freeze, report official train/test exact-duplicate overlap without changing the official test denominator. Optionally calculate a clearly labeled overlap-excluded sensitivity result from the same saved predictions. This public benchmark does not establish customer-disjoint or HSBC-specific performance.

## 5. Preprocessing and bounded experiments

### Why the preprocessing was revised

The first version named three techniques and one comparison (with/without lemmatization). The training audit showed that the generic defaults would silently damage the signals this task depends on:

- 15% of training requests contain an `n't` contraction (`didn't`, `don't`, `can't`, `hasn't`, ...), plus 64 apostrophe-less forms (`dont`, `didnt`) and 12 `cant`. scikit-learn's default `token_pattern` keeps only tokens of two or more word characters, so `can't` becomes `can` (negation lost) and `didn't` becomes the fragment `didn`.
- About 1,000 requests mention top-ups as `top up`, `top-up`, `top-ups`, or `topup`, which should be one feature.
- scikit-learn's generic English stop-word list contains `not`, `no`, `cannot`, `why`, `how`, `when`, `top`, `up`, `back`, `get`, and `amount`; several are words inside BANKING77 label names. Removing them can turn "Why can't I top up?" into almost nothing.
- Requests contain stray newlines, no-break spaces, double spaces, and `£`/`€`/`$` amounts.
- Minimal pairs separate intents by one content word ("fee for my card payment" vs "fee for my transfer"; "forgot my PIN" vs "forgot my passcode"), so preprocessing must not merge or drop content words.

### Lexical branch: ordered, stateless steps

Every step is a row-wise function of the text with no fitted state, so preprocessed strings can be cached for all development rows without leakage. Only the TF-IDF vocabulary and IDF are learned, inside the pipeline, on the fitting partition. Steps, in order:

1. **Light normalization:** Unicode NFKC; typographic quotes, dashes, and ellipsis mapped to ASCII; zero-width characters removed; whitespace (including newlines and no-break spaces) collapsed.
2. **Negation-preserving contraction handling:** `can't`/`cant`/`cannot` → `can not`; `won't` → `will not`; generic `n't` → ` not`; apostrophe-less negations (`dont`, `didnt`, ...); `'re/'ve/'ll/'m/'d` expanded; the ambiguous `'s` clitic dropped; `topup(s)` → `top up(s)`.
3. **Lemmatization (optional):** spaCy `en_core_web_sm` 3.8.0 with POS tags, run on case-preserved text for better tagging. Expected risk: tense carries intent here ("was charged" vs "will I be charged"), so the effect must be measured rather than assumed.
4. **Case folding.**
5. **Tokenization:** a regex tokenizer that keeps single-character tokens, numbers, and currency symbols, splits hyphenated compounds (`top-up` → `top up`), and drops punctuation. Character n-grams (`char_wb`, 2-5) are evaluated as an alternative granularity that tolerates typos and morphology without a lemmatizer.
6. **Number placeholder (optional):** numeric tokens → `<num>`; currency symbols stay as tokens because currency is intent-relevant (exchange-rate and fiat-currency intents).
7. **Stop-word removal (ablation only, off by default):** either scikit-learn's generic list or that list minus a protected set (negations, wh-words, modals, and every stop word appearing in a canonical label name).

Not used, with reasons: stemming (cruder than lemmatization; same tense risk); spelling correction (only six requests use chat abbreviations such as `u` or `pls`, and a corrector could change domain words); removing numbers entirely (amounts may help "wrong amount" intents, so this is tested only as a placeholder).

**Encoder branch:** MiniLM receives step 1 only, keeping case, punctuation, contractions, and numbers. It has its own WordPiece tokenizer and was trained on natural sentences, so lemmas, removed stop words, or placeholders would move its input away from that distribution; Casanueva et al. likewise feed sentence encoders without extra preprocessing.

### Named ablation variants

| Variant | Adds | Compared with |
| --- | --- | --- |
| L0 | scikit-learn `TfidfVectorizer` defaults on raw text (reference) | none |
| L1 | normalization + case folding + regex tokenizer | L0 |
| L2 | + contraction/negation handling | L1 |
| L3 | + number placeholder | L2 |
| L4 | + spaCy lemmatization | L3 |
| L5 | + stop-word removal with protected words | L3 |
| L6 | + generic scikit-learn stop-word list | L3 |
| char | `char_wb` 2-5 on the L3 string | L3 word features |
| word+char | word 1-2 grams + `char_wb` 2-5 on L3 | L3 word features |

Report each variant with its best tuned setting, a 95% bootstrap interval for validation macro-F1 (1,000 resamples, seed 42), and a paired bootstrap difference from its comparison row. In each resample, macro-F1 averages over the intents present in its answers or predictions (an intent absent from both has no F1 and is not counted as 0). Two models are compared only by the paired difference, never by whether their separate intervals overlap. With 1,000 validation requests (3-18 per intent), differences smaller than the interval are not evidence that a step helps; the report must say so instead of ranking such variants.

Use the same fitting/development assignments for every model. TF-IDF vocabulary, IDF, and any learned feature transformations are fitted inside the training pipeline only. Do not fit them on a combined train/test corpus. Frozen encoder inference may be cached, but held-out embeddings must not become training examples or retrieval-library entries.

| ID | Representation | Classifier | Bounded settings |
| --- | --- | --- | --- |
| C1 | Lexical TF-IDF: word variants L0-L6 × {1-gram, 1-2-gram}, char, word+char | Logistic Regression | `C` in {0.1, 1, 10, 100}. |
| C2 | Same lexical candidates | Linear SVM | Same `C` grid. |
| C3 | Frozen MiniLM, L2-normalized vectors | Logistic Regression | `C` in {0.1, 1, 10, 100}. |
| C4 | Same frozen MiniLM vectors | Linear SVM | Same `C` grid. |

The grid is pre-registered in `configs/experiment.json`: 128 lexical fits and 8 embedding-classifier fits, plus at most one edge-extension fit per family. The `C` grid gained 100 because unit-norm embeddings and multinomial LR on sparse TF-IDF can need weak regularization; if a family's winner still sits at a grid edge, the same representation is fitted once at 0.01 or 1000 and adopted only if its rounded macro-F1 is strictly higher. TF-IDF settings are fixed (`min_df=1`, L2 norm, smoothed IDF, no sublinear TF). Use no class weighting initially; report class frequencies. Record solver convergence and elapsed time. These small searches replace the EEG nested Optuna studies.

Within a family, the winner has the highest validation macro-F1; ties after rounding to four decimals prefer fewer TF-IDF features, then smaller `C`, then grid order. Primary model-selection metric across families: validation macro-F1 over the fixed 77-label list. Tie rule: macro-F1 rounded to four decimals, then lower median warm batch-one latency on the same machine, then C1/C2/C3/C4 order. Keep seed 42 fixed; do not search for a favorable seed. Freeze the deployed classifier before opening test results. Score all four prespecified configurations once for the report; the test table must not select a different deployment model.

If DistilBERT is selected as the extension: maximum length 128 after inspecting training token lengths, batch size 16, AdamW weight decay 0.01, learning rate candidates {2e-5, 5e-5}, at most five epochs, and validation macro-F1 checkpoint selection with patience two. These are proposed starting settings, not a promise to reproduce the paper. Run a single-batch smoke check first; adjust batch size for memory before comparative runs, then record and freeze it. Use `[B, 77]` raw logits, integer class targets, and cross-entropy. Never flatten class logits or apply sigmoid. Do not assume classifier hidden states are suitable retrieval embeddings; keep MiniLM retrieval fixed. Name this arm C5: it joins deployment selection under the same split and metric rule only if completed before freeze, with last priority in the final tie-break order. A few-shot SetFit extension remains a separate learning-curve analysis rather than an implicitly comparable full-data candidate.

## 6. Retrieval, routing, and the interface

### Retrieval

R1: fitting-partition TF-IDF vectors ranked by cosine similarity.

R2: fitting-partition MiniLM vectors ranked by cosine similarity, stored in Qdrant for the application. Keep a direct exact-cosine implementation as the correctness reference; configure exact search for this small benchmark or verify Qdrant's top-five agreement. Persist distance settings, embedding revision, and index membership.

Return five requests with original text, row ID, recorded category, and similarity score. A similarity score is not a probability of duplicate meaning or shared resolution. Search the whole library; no predicted-category filter in the primary system. Tie scores break by row ID. The current request must not be inserted before retrieval, and development/test requests never enter the benchmark index.

For development, two teammates judge results for 15 model-validation queries using a written rubric. Select the retrieval method by mean human-judged Precision@5; ties prefer MiniLM for the planned semantic-search demo. Freeze the method before final testing and report both retrieval methods.

For final human evaluation, preselect one official test query per intent using seed 42 after freeze. Pool the top five from R1 and R2, deduplicate query-result pairs, and shuffle the presentation. Two raters independently grade at most 770 pairs with system names and category labels hidden:

- 0: unrelated banking issue.
- 1: related topic but a different specific question or operational problem.
- 2: sufficiently similar specific request to help a support agent understand this case.

Resolve disagreements after saving both original judgments. Primary retrieval result: Precision@5 with relevance defined as grade 2. Report agreement and representative errors. Same-intent@5 across all test queries is a secondary label-consistency proxy, not a substitute for human relevance. Do not claim a measured reduction in support handling time without a user study.

### Routing

Create a versioned `intent_to_queue` mapping covering each canonical intent exactly once. Start with proposed queues such as Cards, Transfers, Cash/ATM, Top-ups, Account/Identity, and Disputes; determine the final inventory by inspecting fitting-partition examples. All names are illustrative project queues. The user's banking experience helps review the taxonomy, without importing private bank data.

Keep intent prediction and queue assignment visible separately. Do not route from label names alone: inspect training examples for ambiguous names. Evaluate top-one intent accuracy and queue accuracy, since a wrong intent can still map to the correct queue.

For the chosen classifier, use the top-one minus top-two score margin as an uncertainty signal. On routing-policy validation, choose the threshold with greatest coverage subject to at most 5% observed intent errors and at least 100 automatically routed requests; ties prefer the larger threshold. If none qualifies, retain category suggestions and route all requests to review. These are demonstration criteria, not a guaranteed operational error rate. Freeze the threshold before testing and report test coverage and error rate among routed requests.

LR probability estimates and SVM decision scores must be labeled appropriately; do not convert SVM margins into percentages or claim calibration. A calibrated-probability feature would need a separate training-only calibration procedure and is optional. Review is an application state, not an invented 78th dataset intent. Unknown and multi-intent messages remain a stated limitation of this single-label benchmark.

### Minimal application

```text
Streamlit request form
        |
     FastAPI
        |-- selected classifier --> intent --> queue mapping / review rule
        |-- retrieval service ---> five similar requests
        `-- correction store ----> separate local SQLite database
```

Use Qdrant for vector search and SQLite for local correction records. FastAPI exposes health, prediction, and feedback operations; the prediction response includes model/config version, top candidates, score type, queue/review status, retrieved IDs/text/categories/scores, and elapsed time. Load models once at startup. Reject blank input and report length-limit handling clearly. Package the working demo with Docker Compose in week 4. Pin dependencies after a tested installation; do not invent version pins in this plan. Implementation reference: [Qdrant local quickstart](https://qdrant.tech/documentation/quickstart/).

PostgreSQL/Supabase can replace SQLite if the team finishes early. Pinecone is an alternative to Qdrant, not a second required vector store. Redis is outside the initial scope. Keep paid APIs and cloud deployment outside the committed project.

## 7. Evaluation and reporting contract

Classification results: accuracy, macro precision/recall/F1, per-intent support and recall, full confusion matrix, and a readable figure of the largest confusions. Include top-three accuracy as a secondary agent-assistance metric. All 3,080 official test inputs remain in the denominator; invalid/missing predictions count as errors. Report failures separately rather than silently excluding them.

Efficiency results: fit time, embedding/index build time, artifact size, and warm batch-one latency p50/p95 measured over the same fixed validation inputs on the same hardware. Report end-to-end API latency separately from model prediction latency; precomputing embeddings moves work to an offline stage rather than eliminating it. Freeze warmup and repetition counts before measurement.

Required robustness slice: fit an auxiliary TF-IDF vocabulary on fitting texts and build one concatenated text profile per intent. After test predictions are locked, select the five test requests per true intent least similar to that intent's fitting profile. Report accuracy and macro-F1 on the resulting 385 requests. True test labels are used only for offline slice scoring. Call this a **prespecified lexical-difficulty slice inspired by Qi et al.**, not an independent test set or exact reproduction; lexical rarity alone does not establish semantic or real-world robustness. Freeze the tokenization and selection rule before testing; following Qi et al., use a whitespace tokenizer that ignores punctuation for this auxiliary profile, independent of the C1/C2 preprocessing choice, so the slice does not favour the selected lexical pipeline. [Paper's evaluation design](https://arxiv.org/html/2012.03929v2)

Optional few-shot experiment: only if chosen in week 1, compare the two LR representations using 5 and 30 fitting examples per intent. Use nested subsets and three fixed sampling seeds, the same validation queries, and no held-out retrieval examples. Fit each lexical vocabulary/IDF and classifier using only that sampled subset; frozen pretrained embeddings remain unchanged. If SetFit is the chosen extension, its supervised pairs also come only from the sampled subset. Report it as a development learning curve. Do not add an unbounded model search or combine it with every other extension.

Save raw predictions with row ID, label, prediction, class ordering, scores, queue, review decision, and retrieved IDs. Save configuration hashes, data/split manifests, label/queue mapping, model/tokenizer revisions, dependencies, training history, and source version. Save TF-IDF and classifier together. Save cached embeddings with row-order checksums. No final score is reported until it has been recomputed from persisted predictions.

Report factual observations, including when TF-IDF wins. Published benchmark exposure is possible for pretrained encoders; document available training-data disclosures. Results support this benchmark and prototype, not verified HSBC deployment performance.

## 8. Reusing the EEG work responsibly

Inspected references: [model_lib.py](model_lib.py), [training_vast30OctStandardCNN.ipynb](training_vast30OctStandardCNN.ipynb), [training.py](training.py), and [training_cnn_tuning.py](training_cnn_tuning.py). Reuse selected patterns in a new NLP project; keep the EEG sources intact.

| EEG knowledge/code | BANKING77 use |
| --- | --- |
| Frozen CBraMod/REVE encoder and classifier | Frozen text encoder and supervised intent classifier; none of the EEG weights transfer. |
| Editable variables and direct notebook function calls | A new notebook controls data preparation, bounded experiments, evaluation, and optional fine-tuning. |
| Seed/config/result persistence | Reuse the pattern and small generic helpers such as `save_json`; replace EEG-specific metadata. |
| Train-only normalization discipline | Train-only TF-IDF fitting and any learned embedding adaptation. |
| Early stopping and saved best model | Optional DistilBERT validation macro-F1 checkpoint selection. |
| Participant-disjoint reasoning | Duplicate-text-aware development partitions; no fictional customer groups. |
| Confusion matrices and held-out predictions | Multiclass and retrieval error analysis. |

Do not copy the selected notebook's training loop verbatim. Cell 5 uses binary sigmoid metrics, flattened logits, float labels, unconditional CUDA AMP, and equally averaged batch losses. Its Optuna objective selects only the first grouped split with `next(...)`. Those assumptions do not implement this NLP protocol. If a neural loop is needed, use integer token IDs/targets, attention masks, 77 logits, conditional AMP, sample-weighted epoch loss, and the explicit development split. The existing EEG image Transformer is not a pretrained language model.

MNE filtering, channel normalization, spectrogram augmentation, electrode positions, participant logit averaging, and EEG SHAP machinery are not NLP dependencies. Neither EEG nested CV nor the EEG full-data SHAP refit stage is required here. Text explanations can begin with TF-IDF feature contributions and retrieved examples; retrieved examples are contextual evidence, not a faithful explanation of the classifier's internal computation.

## 9. Proposed independent project structure

The following is a future layout, not files created by this planning task. Use a separate project directory/repository; a `banking77/` subdirectory is acceptable during development. Imports must not download data, fit models, or start services.

```text
banking77/
  README.md
  requirements.txt
  notebooks/banking77_experiments.ipynb
  banking_nlp/
    data.py              # source provenance, label mapping, duplicate audit, splits
    preprocessing.py     # named lexical variants L0-L6, encoder light normalization
    models.py            # lexical/embedding classifiers; optional text fine-tuning
    experiments.py       # prepare / lexical / embeddings / select / gated final stages
    evaluation.py        # multiclass, bootstrap, queue, retrieval, difficulty-slice metrics
    retrieval.py         # exact reference search and Qdrant adapter
    routing.py           # queue mapping, margin score, review policy
    artifacts.py         # manifests, hashes, model/prediction persistence
    api.py               # inference and feedback operations
  app.py                 # Streamlit interface
  configs/               # pre-registered experiment.json, reviewed queue taxonomy
  artifacts/             # generated models, manifests, embeddings, predictions
  report/                # experiment tables, figures, report, contribution table
  tests/                 # only focused contract and integration checks
  Dockerfile
  compose.yaml
```

Useful checks during implementation: all 77 labels round-trip correctly; development duplicate groups are disjoint; preprocessing cannot fit on held-out data; saved pipeline reload reproduces predictions; retrieval IDs belong only to the fitting library; a query cannot retrieve itself; every intent has one queue; inference imports have no side effects; blank requests are handled; a neural extension preserves token/label dtypes and `[B,77]` outputs. No full training is required merely to check these contracts.

## 10. Four-week work allocation and gates

Planning estimate: 8-12 hours per person per week, or 96-144 total person-hours. These are effort estimates, not measured training times. Use relative weeks; the actual course deadline remains on the LMS.

| Week | Member A: data/evaluation, suggested user role | Member B: modeling | Member C: application/retrieval | Exit gate |
| --- | --- | --- | --- | --- |
| 1 | Dataset audit, frozen manifests, training-only taxonomy review, preprocessing rationale | C1/C2 working, multiclass metrics, define bounded search | Request/result mockup, API contract, direct retrieval prototype | Data checks pass; one reproducible baseline; queue draft; extension choice recorded. |
| 2 | Verify all comparisons use identical splits; define retrieval rubric and difficulty slice | C3/C4, all required validation results; optional extension only if core is complete | Qdrant and R1/R2 parity; integrate prediction and similar requests | Four classifier configurations and both retrievers work; candidate selection evidence saved. |
| 3 | Development relevance ratings, routing-policy evaluation, draft errors and report | Freeze selected classifier, optional extension completed or dropped, save artifacts | Review controls, correction store, API integration, demo rehearsal | Model, retrieval, queue mapping, thresholds, and final scoring rules locked. |
| 4 | Final retrieval ratings, tables/figures, report and contribution table | One final scoring pass per prespecified system; verify saved predictions | Docker packaging, startup instructions, demo rehearsal | Complete submission and reproducible local demonstration. |

All members review error analysis and can explain the final system. Member A and B perform the independent relevance judgments; Member C prepares blinded rating sheets and handles integration. Track meetings and actual contributions for the assignment; do not invent fixed contribution percentages in advance.

Stop rules: if baselines/data checks are incomplete after week 1, drop all model extensions. If the full local API does not work by the end of week 2, prioritize integration over new experiments. If Qdrant blocks progress, use verified exact search to keep the demo working and document Qdrant as deferred. After freeze, UI fixes may continue; prediction changes require a new recorded experiment and cannot be described as untouched testing after inspecting test errors.

## 11. Assignment fit and completion checklist

Source: [course assignment](mlAssignments_v1.1_english.pdf), previously reviewed in this conversation. The report must describe the data, pipeline, experiments, results, and errors; the demo supports the scientific work.

| Requirement | Planned evidence |
| --- | --- |
| Data collection/description | Public source, revision, sizes, class distribution, example messages, duplicate audit. |
| At least three preprocessing techniques | Normalization, negation-preserving contraction handling, regex and character n-gram tokenization, number placeholders, lemmatization, and stop-word removal (evaluated and reported, off by default unless validation supports it), each motivated by the audit and measured in the L0-L6/char ablation. |
| At least two representations | TF-IDF and pretrained sentence embeddings. |
| At least two models | Logistic Regression and linear SVM; optional DistilBERT separately labeled. |
| Evaluation and analysis | Macro metrics, confusion matrix, model comparisons, retrieval judgments, routing coverage/errors, difficulty slice. |
| Source code and report | Reproducible notebook/modules, saved outputs, related work, limitations, contribution table. |
| Optional demonstration | Local request form, routing decision, similar requests, correction recording. |

Definition of done:

- [ ] All required model and representation comparisons have actual saved results.
- [ ] All 77 canonical intents map correctly through the model, API, and queue taxonomy.
- [ ] The official test set did not select preprocessing, models, thresholds, examples, or retrieval settings.
- [ ] The two retrieval methods are evaluated independently of intent classification.
- [ ] The final retrieval library and model match the recorded manifests.
- [ ] Human relevance judgments and disagreement handling are documented.
- [ ] Classification, queue accuracy, automatic-routing coverage, and retrieval relevance are clearly distinguished.
- [ ] Reported test metrics can be recomputed from persisted predictions, including failures.
- [ ] A fresh local environment can load the artifacts and demonstrate the complete workflow.
- [ ] The report cites the sources and separates our measurements from published benchmarks.

No target accuracy is a condition of completion. A trustworthy comparison and working application are the deliverables. Done so far: the data audit, frozen splits, preprocessing ablation, and C1/C2 candidates (`artifacts/dev_results/`). Next milestone: run `python -m banking_nlp.experiments embeddings` on a machine that can reach Hugging Face, pin the recorded MiniLM revision in `configs/experiment.json`, then run `select` to freeze the deployment model before any routing-threshold or test work.