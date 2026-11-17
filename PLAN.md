# BANKING77: customer request classification, routing, and similar-request retrieval

> **14 November 2026:** [SPEC.md](SPEC.md) is now the current two-phase implementation specification. It selects React/TypeScript, FastAPI, PostgreSQL for application records, Qdrant for vector retrieval and local Docker Compose delivery, and updates the research priorities, evaluation protocol and evidence status. The material below is retained as historical research planning; its older stack, mandatory extensions and statements that no transformer test run exists are superseded. [CV_PROJECT.tex](CV_PROJECT.tex) contains the current in-progress CV entry.

Prepared: 3 November 2026. Revised: 6 November 2026 to use native Hugging Face datasets and simple DistilBERT/RoBERTa training.

Status: EDA and C1–C4 load BANKING77 directly with `load_dataset("PolyAI/banking77")`. Native stratified splits contain 9,002 fitting and 1,001 model-validation requests. `training_transformer.ipynb` uses Hugging Face tokenizers/backbones, explicit `RoBERTa` and `DistilBERT` classes, and a PyTorch training loop, with optional 5/30 examples per intent. Full neural training and official test scoring have not run. The project has three people, four weeks, public data, and a local demo; the team lead owns the transformer/LoRA research.

Historical validation results on the previous group-based CSV split from [comparison.csv](artifacts/training/comparison.csv), using the settings recorded in [run_info.json](artifacts/training/run_info.json):

| System | Validation accuracy | Validation macro-F1 |
| --- | --- | --- |
| C1: TF-IDF + Logistic Regression | 90.91% | 0.9102 |
| C2: TF-IDF + linear SVM | 91.01% | 0.9095 |
| C3: frozen MiniLM + Logistic Regression | 93.31% | 0.9365 |
| C4: frozen MiniLM + linear SVM | 92.81% | 0.9306 |

C3 led that earlier experiment. The table above uses a different split and must not be compared directly with new native-split runs. New C1–C4 outputs are in `artifacts/training_hf`; keep all compared models on the same native split. These are development results, not official test scores. Lemmatization, near-duplicate auditing, routing, retrieval evaluation, and the transformer experiments remain outstanding.

## 1. Recommended project and boundaries

Build an agent-facing support workbench that accepts an English banking request, predicts one of 77 intents, assigns a proposed support queue or flags the request for review, and shows five related requests with their recorded categories. A reviewer can correct the category. Corrections are stored separately and do not automatically change the evaluated model or retrieval library.

Research question: **How do lexical and pretrained sentence representations compare for fine-grained banking intent classification, similar-request retrieval, and routing with an option to request human review?**

Team-lead research question: **On the same RoBERTa-base checkpoint and BANKING77 splits, how does LoRA compare with full fine-tuning in macro-F1, trainable parameters, memory use, and elapsed training time?** Llama is a secondary comparison across model families, not evidence that decoder architectures or LoRA are inherently better.

The core project uses public BANKING77 requests as a simulated historical library. Its records provide text and intent, not queue assignments, timestamps, resolutions, or customer IDs. Queue definitions are our project design. BANKING77 evaluates consumer banking intents; HSBC employee IT incidents would require different data. Dataset facts and attribution: [official dataset card](https://huggingface.co/datasets/PolyAI/banking77).

Required scope:

- Four classifier configurations: TF-IDF and frozen MiniLM sentence embeddings, each with Logistic Regression and linear SVM.
- First fine-tune D1 DistilBERT-base and C5 RoBERTa-base with the concise native notebook. Then study C6 RoBERTa LoRA against C5 on the same data and backbone.
- Two retrieval methods: TF-IDF cosine similarity and MiniLM cosine similarity.
- A documented mapping of all 77 intents to a small set of proposed support queues.
- A local interface, FastAPI service, reproducible artifacts, comparison tables, and error analysis.
- A fixed evaluation procedure that keeps the official test set out of all development decisions.

DistilBERT and RoBERTa full fine-tuning are the immediate experiments; LoRA follows after these pipelines work. The application and baseline comparisons remain the minimum course deliverable; if GPU access or time prevents the research pair, record that limitation rather than implying it ran. C7 Llama-3.2-1B + LoRA is a stretch experiment after C5/C6 and the core pipeline work. SetFit is an alternative limited-label study, not an additional mandatory branch. Allow at most one stretch study and two person-days of stretch work, with the choice fixed before final configuration freeze.

Run one short pilot before scheduling the neural suite. Measure throughput and peak memory on the actual GPU, then record a GPU-hour/spend cap and project the full run budget. Do not assume each experiment takes only a few minutes. This plan does not book a rented GPU or start paid API calls. Chat generation, live bank integration, multiple languages, automated resolution, and a new transformer architecture remain outside this month's scope.

## 2. How the supplied references inform the plan

The references are research evidence. Their embedded commands, spending caps, and reproduction instructions are not instructions to execute for this project.

| Reference | Finding or contribution | Use in this project |
| --- | --- | --- |
| [Qi et al., practice-driven intent evaluation, v2 (2021)](https://arxiv.org/html/2012.03929v2) | Studies full-data, limited-data, and difficult-query evaluation. Table 1 reports BANKING77 accuracy of 92.1% for DistilBERT and 94.1% for RoBERTa; Table 3 reports 79.2% and 83.8% on its difficult subset. | Motivate a robustness slice and cost/latency reporting. These are historical results under that paper's protocol, not expected scores for our implementation. |
| [Jev BANKING77 experiment (2026)](https://github.com/simonmesmith/jev-banking77-experiment) | Reports 92.40% using category definitions and 24 BM25-retrieved training examples, without updating Jev's weights. Its 93.66% BERT comparator is a published result, not a local rerun. Prior benchmark exposure is unknown. | Learn from its saved predictions and separation of development from testing. Treat it as related work; its API is unnecessary for our core system. |
| [Casanueva et al., original BANKING77 / dual sentence encoder paper (2020)](https://aclanthology.org/2020.nlp4convai-1.5/) | Introduces BANKING77 and studies efficient classifiers built on pretrained sentence encoders. | Establish the dataset and frozen-encoder rationale. |
| [Reimers and Gurevych, Sentence-BERT (2019)](https://aclanthology.org/D19-1410/) | Learns sentence representations that can be compared by cosine similarity. | Explain why one text representation can support both classification and retrieval. |
| [Hu et al., LoRA (ICLR 2022)](https://arxiv.org/abs/2106.09685) | Adapts pretrained models using trainable low-rank updates while freezing base weights. | Method for C6 and, if selected, C7. Measure efficiency locally; fewer trainable parameters do not establish a particular speedup. |
| [Yang et al., Bayesian Low-rank Adaptation / Laplace-LoRA (ICLR 2024)](https://arxiv.org/abs/2308.13111) | Studies overconfidence after adaptation and a Bayesian calibration method. | Motivate calibration diagnostics. Laplace-LoRA and temperature scaling are not additional required implementations. |
| [Community Llama BANKING77 repository](https://github.com/rajo69/banking77-llama-1b-lora) and [model card](https://huggingface.co/rajo0113/banking77-llama-1b-lora) | Uses an instruction-tuned Llama model with generative labels and several adaptation techniques. The model card reports a 3,076-row evaluation split and its use for checkpoint selection. | Implementation reference only. Its data variant, output method, and test-informed checkpoint selection differ from our protocol; its reported score is not a controlled benchmark for C7. |

Keep published results in a separate related-work table. Our train/development allocation differs from papers that fit on all official training examples. Do not claim a reproduced comparison, current state of the art, superiority, or equivalent performance from these numbers. The Jev repository's reported API costs are historical estimates, not a budget quote for a new run.

Do not carry over unsupported claims that CNNs/LSTMs cannot be competitive, that no relevant peer-reviewed study exists, or that a larger decoder must beat an encoder. Excluding a model here is a scope decision. Record exact source revisions when reproducing any external setup.

## 3. Foundation model, custom model, or API?

**Decision: retain the four baselines, run DistilBERT/RoBERTa through native Hugging Face APIs, then compare RoBERTa full fine-tuning with LoRA.** Using the same backbone makes the adaptation comparison interpretable. Comparisons against MiniLM or Llama also change model size, architecture, and pretraining, so they cannot isolate adaptation alone.

| Option | Decision | Reason |
| --- | --- | --- |
| TF-IDF + Logistic Regression / linear SVM | Required | Cheap, interpretable lexical controls and two distinct classifier families. |
| `sentence-transformers/all-MiniLM-L6-v2` + the same classifiers | Required | One frozen representation supports classification and search. Cache embeddings in stable row order. |
| D1: `distilbert/distilbert-base-uncased`, full fine-tuning | Immediate experiment | Smaller pretrained encoder; same split, label budgets, and training recipe as C5. |
| C5: `FacebookAI/roberta-base`, full fine-tuning, new 77-class head | Main research comparison | Train the backbone and head; this is the reference for C6. |
| C6: the same RoBERTa-base, LoRA plus fully trained 77-class head | Main research comparison | Keep backbone revision, data, head initialization per seed, and evaluation protocol aligned with C5. |
| C7: `meta-llama/Llama-3.2-1B`, LoRA plus 77-class classification head | Stretch; not required | Compare a larger decoder-based classifier with the completed encoder systems, including memory and latency costs. |
| Llama label generation / instruction prompting | Defer | Adds prompt, output-parsing, and class-score decisions to a project whose routing policy expects a score for every intent. |
| SetFit on the same MiniLM backbone | Alternative to C7 if limited-label learning is chosen | Compare against frozen MiniLM and TF-IDF on identical small training subsets; bound pair sampling. |
| Custom CNN/LSTM/transformer from scratch | Defer | Adds a separate architecture study and implementation effort without answering the chosen adaptation question. It is not required by the assignment. |
| Jev or another hosted decision/LLM API | Related work by default | Adds a service dependency, usage accounting, and unknown pretraining exposure. A paid reproduction is a separate decision. |

MiniLM produces 384-dimensional vectors; record its exact model revision, maximum input length, pooling, and embedding normalization. Start with the model's documented tokenizer and pooling. [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

For C5/C6, use a general pretrained RoBERTa checkpoint and initialize a new intent head; do not start from someone else's BANKING77-trained model. Pin the model/tokenizer revision. DistilBERT is a separate encoder comparison requested by the user, not a substitute for either member of the later RoBERTa/LoRA pair. [RoBERTa model card](https://huggingface.co/FacebookAI/roberta-base)

For C7, use the base Llama checkpoint named above with a sequence-classification head; an instruction-tuned checkpoint would define a different experiment. The official Hugging Face model is gated and has the Llama 3.2 Community License. Model access must be available before C7 is scheduled; keep credentials out of notebooks and source control. If access is unavailable, defer C7 rather than silently switching models. [Llama model card](https://huggingface.co/meta-llama/Llama-3.2-1B)

SetFit provides prompt-free few-shot training of sentence transformers; it is relevant only if limited-label learning becomes the chosen extension. [SetFit documentation](https://huggingface.co/docs/setfit/index)

## 4. Data contract and frozen evaluation protocol

Use the official 10,003 training and 3,080 test examples, with the original 77 label names. Read the label mapping from dataset metadata; do not infer IDs from a separately ordered list. Preserve original capitalization and punctuation in canonical labels, while allowing friendly display names. Attribute BANKING77 under its published CC-BY-4.0 license. [Dataset source](https://huggingface.co/datasets/PolyAI/banking77)

Load the native Hugging Face `DatasetDict` in each notebook. Read names from `ds["train"].features["label"].names`; keep its integer labels. Pandas is only for EDA and baseline result tables. No `data/processed` export, CSV reload, or separate `labels.json` is required.

Use these same calls in every notebook:

```python
ds = load_dataset("PolyAI/banking77")
split = ds["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
train_ds, valid_ds = split["train"], split["test"]
```

All 77 intents occur in each partition, with at least 30 fitting examples per intent. The former 80/10/10 allocation left one intent with only 28 fitting examples, so it could not support an exact 30-shot comparison. Keep the split seed fixed when changing training/sampling seeds. Native splitting is row-based: the current audit finds one normalized text shared by fitting/model-validation. Report this limitation; do not describe these partitions as duplicate-group-disjoint. Semantic near duplicates are also not excluded. Earlier CSV-based results remain historical; never combine their scores with native-split comparisons.

Keep the official test set out of preprocessing fitting, model selection, few-shot sampling, retrieval development, and threshold selection. Hugging Face caches downloaded data itself. Record the dataset fingerprint and model revision with completed results.

Current data usage:

| Partition | Allowed use |
| --- | --- |
| Fitting, about 90% of official train | TF-IDF fitting, supervised model training, sentence-encoder fine-tuning if selected, and the retrieval library. |
| Model validation, about 10% | Hyperparameter/model selection, epoch choice, preprocessing ablations, and development retrieval judgments. |
| Routing calibration, deferred | Before implementing automatic review, reserve independent training-origin data and retrain consistently, or specify an out-of-fold procedure. Do not tune and evaluate thresholds on the same validation rows. |
| Official test | Final scores for the prespecified systems, locked robustness analysis, and final retrieval judgments. |

**Default final procedure: do not refit before testing.** Evaluate the selected development-trained artifacts and the unchanged fitting-partition retrieval library. This keeps validation and final scoring separate. No routing threshold has been calibrated under the current split. Report the actual fitting count, so published full-training scores are not treated as controlled comparisons. A later all-training deployment refit is a separate artifact requiring its own calibration/validation procedure; do not attach the earlier measured score to it.

After configuration freeze, report official train/test exact-duplicate overlap without changing the official test denominator. Optionally calculate a clearly labeled overlap-excluded sensitivity result from the same saved predictions. This public benchmark does not establish customer-disjoint or HSBC-specific performance.

## 5. Preprocessing and bounded experiments

Keep the current light `clean_text()` as the default: Unicode/whitespace normalization with case, punctuation, numbers, and negation preserved. Tokenization belongs to the chosen representation. To complete the assignment's third preprocessing technique, add lemmatization as a separate TF-IDF ablation on model validation, not as a replacement for `text_clean`. Transformer models always receive the light text and their own tokenizer. Show a few examples and explain the different choices.

Use the same fitting/development assignments for every model. TF-IDF vocabulary, IDF, and any learned feature transformations are fitted inside the training pipeline only. Do not fit them on a combined train/test corpus. Frozen encoder inference may be cached, but held-out embeddings must not become training examples or retrieval-library entries.

| ID | Representation | Classifier | Bounded settings |
| --- | --- | --- | --- |
| C1 | Word TF-IDF | Logistic Regression | Word 1-grams vs 1-2 grams; `C` in {0.1, 1, 10}. |
| C2 | Word TF-IDF | Linear SVM | Same feature candidates and `C` grid. |
| C3 | Frozen MiniLM, L2-normalized vectors | Logistic Regression | `C` in {0.1, 1, 10}. |
| C4 | Same frozen MiniLM vectors | Linear SVM | `C` in {0.1, 1, 10}. |

Use no class weighting initially; report class frequencies. Evaluate lemmatization on C1/C2, giving at most 24 lexical fits and six embedding-classifier fits. Record solver convergence and elapsed time. These small searches replace the EEG nested Optuna studies.

Primary model-selection metric: validation macro-F1 over the fixed 77-label list. Tie rule: macro-F1 rounded to four decimals, then lower median warm batch-one latency on the same deployment machine, then C1–C7 order. Within a classical family, remaining ties prefer lower `C`, then the n-gram setting. The seed-42 run is the deployment candidate for each neural arm; do not choose the best seed. Freeze the deployed classifier before opening test results. Score each completed, prespecified system once for the report; the test table must not select a different deployment model. SetFit's separate few-shot results do not join this full-data selection.

### Transformer and LoRA experiments

The notebook uses `AutoTokenizer`, `DataCollatorWithPadding`, and PyTorch `DataLoader`. Select `distilbert/distilbert-base-uncased` or `FacebookAI/roberta-base` with one checkpoint variable. `model_lib.py` defines two `nn.Module` classes: `DistilBERT` selects `[CLS]` and uses a dense/ReLU/dropout/linear head; `RoBERTa` selects `<s>` and uses a dropout/dense/tanh/dropout/linear head. Both return 77 logits and fine-tune every backbone/head parameter. The notebook explicitly shows cross-entropy, backpropagation, gradient clipping, AdamW, scheduling, validation, and checkpoint selection. Softmax is applied when displaying inference probabilities. [Native PyTorch training reference](https://huggingface.co/docs/transformers/v4.40.2/training)

Starting recipe from [Qi et al., §4](https://arxiv.org/html/2012.03929v2): AdamW, learning rate `4e-5`, weight decay `0.01`, batch size 32, maximum length 128, linear decay, 10% warmup. Use dynamic padding and select the best validation macro-F1 checkpoint. The scheduler warms up for `ceil(0.1 * epochs * len(train_loader))` updates. Use BF16 autocasting on supported CUDA GPUs; otherwise FP32. The current loop uses batch size 32 without gradient accumulation. The first configuration cell exposes checkpoint, label budget, seeds, batch size, learning rate, weight decay, epochs, maximum length, warmup ratio, gradient clipping, classifier-head dropout, and BF16. These values feed the loaders, model, optimizer, scheduler, and training loop directly.

| Training budget | Labeled fitting examples | Epochs |
| --- | --- | --- |
| 5 per intent | 385 | 50 |
| 30 per intent | 2,310 | 18 |
| All fitting rows | 9,002 | 5 |

Few-shot subsets are nested at a fixed sampling seed, drawn only from fitting rows, with unchanged validation. These runs measure label efficiency; extra epochs compensate for fewer updates per epoch, not for missing information. Our validation holdout and checkpoint selection differ from the paper's full-data protocol, so this is not an exact reproduction. Start with one full-fitting run per backbone, then run the label-budget comparison if desired. Repeating sampling/training seeds 42/43/44 measures their combined variability; keep the split at seed 42.

After full fine-tuning works, add C6 RoBERTa LoRA (`r=16`, `alpha=32`, dropout `0.1` on query/value projections, head fully trained), with learning rates `{1e-4, 3e-4}`. Use the same full-data epoch cap, effective batch size, split, backbone revision, and paired head initialization as C5; report unequal compute honestly. C7 Llama remains a separate stretch option with a declared pilot/budget, not part of the current notebook.

Before the comparative runs, check that the intended adapter modules and head are trainable, log total/trainable parameter counts, and verify a saved/reloaded checkpoint reproduces predictions. For RoBERTa, explicitly save the full `classifier` head alongside adapters; for Llama, save its `score` head. Confirm names against the loaded architecture rather than assuming a generic target list. If merging adapters for inference, verify merged and unmerged logits agree within the chosen precision before measuring latency. [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora)

If C7 is selected later, specify its learning-rate candidates and budget after a pilot, using the same split and checkpoint-selection rule. Use `AutoModelForSequenceClassification`, not label generation. Configure a pad token and attention masks, and verify that the class logits for an example agree when processed alone and in a padded batch. Use a valid final non-padding token representation. Start without quantization; if the pilot requires QLoRA, record it as C7-QLoRA and document the changed precision/memory setup rather than treating it as the same arm. [Llama implementation documentation](https://huggingface.co/docs/transformers/en/model_doc/llama)

Keep MiniLM retrieval fixed: a fine-tuned classification model is not automatically a suitable retrieval encoder. C5/C6 and optional C7 join deployment selection only when completed, reloadable, and benchmarked on the common deployment machine before configuration freeze.

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

Future application work: the current 90/10 notebooks train intent classifiers only. Add independent routing calibration data (and retrain consistently) before implementing the threshold procedure below; there is no reserved routing split today.

Create a versioned `intent_to_queue` mapping covering each canonical intent exactly once. Start with proposed queues such as Cards, Transfers, Cash/ATM, Top-ups, Account/Identity, and Disputes; determine the final inventory by inspecting fitting-partition examples. All names are illustrative project queues. The user's banking experience helps review the taxonomy, without importing private bank data.

Keep intent prediction and queue assignment visible separately. Do not route from label names alone: inspect training examples for ambiguous names. Evaluate top-one intent accuracy and queue accuracy, since a wrong intent can still map to the correct queue.

For the chosen classifier, use the top-one minus top-two score margin as an uncertainty signal. On routing-policy validation, choose the threshold with greatest coverage subject to at most 5% observed intent errors and at least 100 automatically routed requests; ties prefer the larger threshold. If none qualifies, retain category suggestions and route all requests to review. These are demonstration criteria, not a guaranteed operational error rate. Freeze the threshold before testing and report test coverage and error rate among routed requests.

LR probability estimates, neural softmax scores, and SVM decision scores must be labeled appropriately. Neural logits are converted to softmax scores for the top-two margin; SVM uses its decision values. Thresholds are model-specific: do not transfer the C3 threshold to C5/C6/C7 or interpret a score as a demonstrated operational probability. A calibrated-probability feature would need a separate calibration procedure and is optional. Review is an application state, not an invented 78th dataset intent. Unknown and multi-intent messages remain a stated limitation of this single-label benchmark.

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

For C5/C6 and optional C7, also record total/trainable parameters, peak GPU memory, GPU model, training precision, actual optimizer steps, and GPU-hours. Separate shared model-download/encoding costs from classifier fitting. Repeat the existing latency procedure (20 fixed validation texts, five warmups, three passes) for all finalists on one deployment machine. Keep CPU and GPU timing tables separate; synchronize GPU work when timing it. Record whether LoRA is merged. Do not compare the saved C1–C4 CPU numbers directly with new GPU numbers or assume LoRA is faster solely because it trains fewer parameters.

For the neural research pair, add diagnostic negative log-likelihood and top-label expected calibration error using 15 fixed equal-width confidence bins on `[0, 1]`; record the bin convention. Compute them on model validation and, after prediction freeze, official test predictions. This measures calibration without fitting a calibration model. Temperature scaling and Laplace-LoRA are deferred: introducing either needs an independent calibration allocation or an out-of-fold procedure, not fitting a temperature and claiming independent calibration quality on those same rows. Independent routing calibration remains future work; the current notebook has no routing holdout.

Required robustness slice: fit an auxiliary TF-IDF vocabulary on fitting texts and build one concatenated text profile per intent. After test predictions are locked, select the five test requests per true intent least similar to that intent's fitting profile. Report accuracy and macro-F1 on the resulting 385 requests. True test labels are used only for offline slice scoring. Call this a **prespecified lexical-difficulty slice inspired by Qi et al.**, not an independent test set or exact reproduction; lexical rarity alone does not establish semantic or real-world robustness. Freeze the tokenization and selection rule before testing. [Paper's evaluation design](https://arxiv.org/html/2012.03929v2)

Optional SetFit study, chosen instead of C7: compare TF-IDF + LR, frozen MiniLM + LR, and SetFit on the same MiniLM backbone using 5 and 30 fitting examples per intent. Use nested subsets and sampling seeds 42/43/44, fixed training seed 42, the same model-validation queries, and no held-out pairs. Fit each lexical vocabulary/IDF and classifier only on its sampled subset. Bound SetFit to at most 10,000 sampled pairs and 200 encoder optimizer steps per fit; log the actual sampling settings and steps. Use one prespecified configuration per method, not a full-data winner's tuned settings presented as a few-shot-only experiment. Report a separate development learning curve and label the sampling variability. Keep this study outside the full-data deployment ranking.

Save raw predictions with row ID, label, prediction, class ordering, scores, queue, review decision, and retrieved IDs. Save configuration hashes, data/split manifests, label/queue mapping, model/tokenizer revisions, dependencies, training history, and source version. Save TF-IDF and classifier together. Save cached embeddings with row-order checksums. No final score is reported until it has been recomputed from persisted predictions.

Report factual observations, including when TF-IDF wins. Published benchmark exposure is possible for pretrained encoders; document available training-data disclosures. Results support this benchmark and prototype, not verified HSBC deployment performance.

## 8. Implementation style

Keep code concise and notebook-driven, following the user's explicit preference. Match the healthcare project's separation: `model_lib.py` contains architecture classes only; `training.py` holds the baseline workflow helpers, and training notebooks handle experiments and the explicit neural training loop. Reuse generic experiment habits from the EEG work (fixed seeds, saved checkpoints, validation-based selection), not its architectures or preprocessing. No custom framework, broad Optuna study, or new from-scratch architecture is needed for this research question.

Use a standard Transformers/PEFT training implementation with an explicit 77-class head, integer labels, attention masks, and the native dataset splits. A single-batch training smoke check must verify finite loss, gradients in the intended parameters, and `[B, 77]` outputs before each new arm. Text explanations can begin with TF-IDF feature contributions and retrieved examples; retrieved examples provide context, not a faithful explanation of neural internals.

## 9. Project structure

Keep the working files and add only what the next experiment needs. The transformer notebook is implemented; full pretrained neural runs remain to be executed. Imports must not download data, fit models, or start services. Split helpers into more modules only when reuse makes that clearer.

```text
hub-banking-support-ticket-routing/
  README.md
  PLAN.md
  preprocess.ipynb          # EDA, light cleaning, native split demonstration
  training.ipynb            # existing C1–C4; add bounded lexical ablation
  model_lib.py              # classifier architectures only (__init__ and forward)
  training_transformer.ipynb # PyTorch batch loop; DistilBERT/RoBERTa; 5/30/all budgets
  training.py               # baseline construction, fitting, metrics, prediction, latency
  data/processed/           # legacy exports; no longer used by the notebooks
  artifacts/training/       # historical grouped-split results
  artifacts/training_hf/    # baseline outputs using native HF splits
  artifacts/transformer/    # per-model/budget/seed PyTorch checkpoints, tokenizer, history
  app.py / api.py           # planned local demonstration
  retrieval.py / routing.py # planned application functions
  report/                  # comparison tables, figures, and contribution table
```

Useful checks during implementation: all 77 labels round-trip correctly; duplicate overlap is reported; preprocessing cannot fit on held-out data; saved pipeline reload reproduces predictions; retrieval IDs belong only to the fitting library; a query cannot retrieve itself; every intent has one queue; inference imports have no side effects; blank requests are handled; a neural extension preserves token/label dtypes and `[B,77]` outputs. No full training is required merely to check these contracts.

## 10. Four-week work allocation and gates

Planning estimate: 8-12 hours per person per week, or 96-144 total person-hours. These are effort estimates, not measured training times. Use relative weeks; the actual course deadline remains on the LMS.

| Week | Member A: team lead, transformer/LoRA research | Member B: data, baselines, evaluation | Member C: application/retrieval | Exit gate |
| --- | --- | --- | --- | --- |
| 1 | Review RoBERTa/LoRA, pin checkpoint, run smoke checks and a GPU pilot, bound research budget | Verify existing C1–C4 artifacts/splits, close data-audit gaps, add the lexical lemmatization ablation | Request/result mockup, queue draft, direct retrieval prototype | Baselines verified; C5/C6 setup and budget feasible; third preprocessing technique documented. |
| 2 | Finish C5/C6 learning-rate selection and seed repeats; record efficiency and errors | Check metrics and shared partitions; define difficulty slice and retrieval rubric | Qdrant and R1/R2 parity; integrate prediction and similar requests | Research pair completed or limitation recorded; both retrievers work. Decide whether there is capacity for C7 or SetFit. |
| 3 | Finish the single selected stretch study or defer it; write research comparison and help freeze model choice | Development retrieval ratings, routing threshold, saved-prediction checks | Review controls, correction store, API integration, demo rehearsal | Model, retrieval, queue mapping, thresholds, and final scoring rules locked. |
| 4 | Analyze locked final results, document tradeoffs and research contribution | Final scoring for prespecified artifacts, retrieval ratings, tables/figures | Packaging, startup instructions, demo rehearsal | Complete submission and reproducible local demonstration. |

All members review error analysis and can explain the final system. Member A and B perform the independent relevance judgments; Member C prepares blinded rating sheets and handles integration. Track meetings and actual contributions for the assignment; do not invent fixed contribution percentages in advance.

Stop rules: if baseline/data gaps remain after week 1, resolve them before comparative neural training. If C5/C6 are incomplete or the core app is not working by the end of week 2, drop C7/SetFit and finish the research pair plus integration. If the research pair itself exceeds the recorded time/GPU cap, stop expansion and report the completed work without invented results. If Qdrant blocks progress, use verified exact search and document Qdrant as deferred. After freeze, UI fixes may continue; prediction changes require a new recorded experiment and cannot be described as untouched testing after inspecting test errors.

## 11. Assignment fit and completion checklist

Source: [course assignment](mlAssignments_v1.1_english.pdf), previously reviewed in this conversation. The report must describe the data, pipeline, experiments, results, and errors; the demo supports the scientific work.

The assignment explicitly lists Sentence-BERT under representations (§4.3) and BERT under models (§4.4). It does not require a novel architecture or training from scratch. C5/C6 are our chosen research contribution, not additional instructor-mandated models; LoRA is not itself a different base architecture.

| Requirement | Planned evidence |
| --- | --- |
| Data collection/description | Public source, revision, sizes, class distribution, example messages, duplicate audit. |
| At least three preprocessing techniques | Normalization, tokenization, lemmatization, with motivations and a bounded ablation. |
| At least two representations | TF-IDF and pretrained sentence embeddings. |
| At least two models | Logistic Regression and linear SVM cover this minimum; the RoBERTa pair adds a transformer/adaptation study. |
| Evaluation and analysis | Macro metrics, confusion matrix, model comparisons, retrieval judgments, routing coverage/errors, difficulty slice. |
| Source code and report | Reproducible notebook/modules, saved outputs, related work, limitations, contribution table. |
| Optional demonstration | Local request form, routing decision, similar requests, correction recording. |

Definition of done:

- [ ] All required model and representation comparisons have actual saved results.
- [ ] C5/C6 share the recorded backbone, splits, head initialization, and budget protocol; any missing research arm is explicitly documented.
- [ ] Any C7 or SetFit study is separately identified, bounded, and completed before final evaluation decisions are frozen.
- [ ] Three justified preprocessing techniques are implemented, including the separate lexical lemmatization comparison.
- [ ] All 77 canonical intents map correctly through the model, API, and queue taxonomy.
- [ ] The official test set did not select preprocessing, models, thresholds, examples, or retrieval settings.
- [ ] The two retrieval methods are evaluated independently of intent classification.
- [ ] The final retrieval library and model match the recorded manifests.
- [ ] Human relevance judgments and disagreement handling are documented.
- [ ] Classification, queue accuracy, automatic-routing coverage, and retrieval relevance are clearly distinguished.
- [ ] Reported test metrics can be recomputed from persisted predictions, including failures.
- [ ] A fresh local environment can load the artifacts and demonstrate the complete workflow.
- [ ] The report cites the sources and separates our measurements from published benchmarks.

No target accuracy is a condition of completion. A trustworthy comparison and working application are the deliverables. The next experiment is a full DistilBERT run, followed by RoBERTa using the same native notebook and splits. Then compare 5/30/all fitting budgets or add the paired RoBERTa LoRA experiment. Do not add every candidate family at once.
