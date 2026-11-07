# Full results: C1–C4 on model validation, deployment freeze, routing, and the official test set

Run: 7 November 2026 · protocol `configs/experiment.json` hash `f855ee51c3b3` · Windows 11, Python 3.13.12,
scikit-learn 1.9.0, sentence-transformers 6.0.0, torch 2.12.0 (CUDA, RTX 4060 Laptop) ·
`sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
(mean pooling, Normalize module, max 256 tokens; longest training request 98 tokens, none truncated).

Order of operations (each output was written before the next step read anything new):
`prepare` → `lexical` (128 + 1 fits) → `embeddings` (8 + 1 fits) → `select` (freeze, `artifacts/frozen/selection.json`)
→ `routing` (threshold on routing validation, `artifacts/frozen/routing.json`) → `final` (one test pass for all four
families, then scoring from persisted predictions). Development preprocessing details and the ablation:
[`dev_results_lexical.md`](dev_results_lexical.md).

## 1. Model validation (1,000 requests; fitted on 8,003)

| Family | Representation | Classifier, `C` | Macro-F1 [95% CI] | Accuracy [95% CI] | Top-3 | Fit | p50 / p95 batch-one |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | word 1–2 + char_wb 2–5 TF-IDF on L3 (32,369 features) | LR, 100 | **0.923** [0.902, 0.937] | 0.921 [0.903, 0.938] | 0.967 | 10.8 s | 1.18 / 1.42 ms |
| C2 | same | Linear SVM, 1 | 0.909 [0.885, 0.923] | 0.910 [0.891, 0.927] | 0.966 | 2.4 s | 1.19 / 1.77 ms |
| C3 | frozen MiniLM, 384-d, L2-normalized | LR, 100 | 0.918 [0.896, 0.933] | 0.915 [0.897, 0.932] | 0.976 | 3.8 s | 9.52 / 13.35 ms |
| C4 | same | Linear SVM, 1 | 0.914 [0.891, 0.929] | 0.913 [0.896, 0.930] | 0.969 | 8.8 s | 9.31 / 11.80 ms |

C3's winner sat at the grid edge (`C` = 100); the edge rule fitted `C` = 1000 once (0.907) and did not adopt it.
All 138 fits converged. MiniLM latency includes encoding one request on the GPU; encoding all 10,003 development
requests took 2.4 s. Model files: C1/C2 ≈ 20.6 MB each, C3/C4 ≈ 0.1–0.2 MB (plus the 90 MB encoder).

Paired bootstrap on model validation (same 1,000 resamples): C1 − C2 = +0.014 [+0.003, +0.027];
C1 − C3 = +0.004 [−0.012, +0.021]; C1 − C4 = +0.009 [−0.009, +0.029]; C3 − C4 = +0.005 [−0.007, +0.017].
Only the LR-vs-SVM difference on TF-IDF is distinguishable from noise on this partition.

**Frozen deployment choice: C1** (highest rounded validation macro-F1). Ranking: C1, C3, C4, C2.

## 2. Routing threshold (routing validation, 1,000 requests, frozen before test)

Signal: C1 top-1 minus top-2 probability (uncalibrated). Rule: greatest coverage with ≤ 5% observed intent errors
among routed requests and ≥ 100 routed; ties prefer the larger threshold.
Result: threshold **0.4276**; 890 of 1,000 routed (coverage 0.89), error among routed 4.94%; C1 accuracy on the
partition 0.902. The remaining 110 go to review.

## 3. Official test set (3,080 requests, opened once)

| Family | Accuracy | Macro-F1 [95% CI] | Macro-P | Macro-R | Top-3 | Lexical-difficulty slice macro-F1 (n = 385) |
| --- | --- | --- | --- | --- | --- | --- |
| C1 (deployed) | 0.906 | 0.906 [0.894, 0.915] | 0.909 | 0.906 | 0.976 | 0.647 |
| C2 | 0.909 | 0.910 [0.898, 0.919] | 0.912 | 0.909 | 0.972 | 0.653 |
| C3 | 0.920 | 0.920 [0.910, 0.928] | 0.922 | 0.920 | 0.978 | 0.730 |
| C4 | **0.927** | **0.927** [0.916, 0.936] | 0.930 | 0.927 | 0.977 | **0.736** |

No invalid predictions. Train/test overlap: 7 test requests share an exact duplicate key (NFKC, case folding,
whitespace) with official training; 25 do when punctuation is also ignored, the same count the Jev experiment's
README reports for its normalized text. Excluding the 7 changes no macro-F1 by more than 0.0003; excluding the 25
(n = 3,055) lowers each macro-F1 by 0.0005–0.0008 and changes no ranking.

Paired bootstrap on test (1,000 resamples of the 3,080 requests, seed 42, shared by all pairs):
C4 − C1 = +0.021 [+0.012, +0.031]; C3 − C1 = +0.014 [+0.004, +0.024]; C4 − C2 = +0.017 [+0.008, +0.027];
C3 − C2 = +0.010 [+0.001, +0.020]; C4 − C3 = +0.007 [+0.002, +0.013]; C2 − C1 = +0.004 [−0.002, +0.010].
Intervals and differences are written by `score_test` to `artifacts/final/test_ci_and_paired.json`.

Routing on test with the frozen threshold: 2,745 of 3,080 routed (coverage 0.891), error among routed **4.0%**,
335 to review.

Difficulty slice: the five test requests per intent least similar to that intent's fitting-partition TF-IDF profile
(whitespace tokens, punctuation ignored; a prespecified slice inspired by Qi et al., not a reproduction). Every
model loses 19–26 points; the lexical models lose the most (C1 0.906 → 0.647), the MiniLM models least (C4 0.927 → 0.736).

## 4. What the test results say about the development protocol

- **The validation choice did not transfer.** C1 was best on model validation and is the weakest of the four on
  test (−0.017 from validation to test); C4 was third on validation and best on test (+0.013). The C1 figure was
  the maximum over 64 LR fits on the same 1,000 requests, the MiniLM figures over 4 each, so C1 carried the largest
  selection optimism. With 3–18 validation requests per intent, the validation differences among C1, C3 and C4 were
  never distinguishable from noise (Section 1); the rule still had to pick one.
- Under the pre-registered protocol the deployment model stays **C1**; the test table must not be used to switch
  it. A later deployment decision should use a fresh procedure, e.g. repeated stratified-group cross-validation over
  fit + model validation, and would need its own routing threshold.
- The routing rule held up: 4.0% error among routed test requests vs the 5% target, at the same coverage as on
  routing validation.
- Lexical rarity hurts TF-IDF far more than the frozen encoder (slice: 0.647–0.653 vs 0.730–0.736), consistent with
  sentence embeddings generalising beyond surface overlap.

## 5. Context (not comparable)

Casanueva et al. (2020) report 92.81–93.36% accuracy for MLPs on frozen USE/ConveRT encodings and 93.66% for
fine-tuned BERT, trained on all 10,003 requests. Our models were fitted on 8,003 and use different encoders and
classifiers; C4's 92.7% is in the same range but is not a controlled comparison.
