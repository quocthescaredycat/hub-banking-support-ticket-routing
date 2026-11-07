"""Development experiments and the gated final scoring stage.

Stages (callable from the notebook, or ``python -m banking_nlp.experiments <stage>``):

  prepare     download/verify source, audit official training, freeze partitions
  lexical     C1/C2 grid incl. the preprocessing ablation (fit -> model_val)
  embeddings  C3/C4 grid on frozen MiniLM vectors (needs sentence-transformers and
              access to the Hugging Face hub, or a local copy of the model)
  select      latency on one machine + deployment choice -> artifacts/frozen/selection.json
  routing     review threshold for the deployed family on routing_val -> artifacts/frozen/routing.json
  final       predictions for C1-C4 on the official test set, then separate scoring.
              Refuses to run unless selection.json and a matching routing.json exist and
              --i-confirm-freeze is given.

The ``routing_val`` partition is never used for fitting or selection here; it is
reserved for the review threshold of the already-selected classifier (its MiniLM
vectors are cached alongside the other development rows for that later stage).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data as D
from .artifacts import (config_hash, environment_info, load_json, row_order_checksum, save_json,
                        sha256_file)
from .evaluation import (bootstrap_ci, bootstrap_indices, classification_metrics, latency_batch_one,
                         paired_bootstrap_delta, per_class_table, top_confusions)
from .models import (SCORE_TYPE, LexicalRep, class_scores, deployable_embedding_pipeline,
                     deployable_lexical_pipeline, fit_with_diagnostics, make_embedding_model,
                     make_lexical_model, n_features)
from .preprocessing import (CHAR_BASE_VARIANT, LEXICAL_VARIANTS, encoder_input, preprocess_lexical,
                            protected_words)

FAMILIES = {"C1": ("lexical", "lr"), "C2": ("lexical", "svm"),
            "C3": ("embedding", "lr"), "C4": ("embedding", "svm")}


class Paths:
    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()
        self.raw = self.root / "data" / "raw"
        self.cfg = self.root / "configs" / "experiment.json"
        a = self.root / "artifacts"
        self.data = a / "data"
        self.dev = a / "dev_results"
        self.models = a / "models"
        self.emb = a / "embeddings"
        self.pred_val = a / "predictions" / "model_val"
        self.pred_test = a / "predictions" / "test"
        self.frozen = a / "frozen"
        self.final = a / "final"
        self.splits = self.data / "splits.csv"


def load_config(paths: Paths) -> dict:
    cfg = load_json(paths.cfg)
    cfg["_hash"] = config_hash({k: v for k, v in cfg.items() if not k.startswith("_")})
    return cfg


# --------------------------------------------------------------------------------------
# Stage: prepare
# --------------------------------------------------------------------------------------

def prepare(root: str | Path = ".") -> pd.DataFrame:
    """Verify the source, audit official training, and freeze the partitions.

    If ``splits.csv`` already exists it is loaded and re-checked, never recomputed,
    so every later stage uses the exact saved assignment.
    """
    P = Paths(root)
    cfg = load_config(P)
    manifest = D.download_source(P.raw)
    labels = D.load_label_names(P.raw)
    save_json(manifest, P.data / "source_manifest.json")
    save_json({"label_names": labels, "label_to_id": {n: i for i, n in enumerate(labels)},
               "source": "categories.json order at the pinned commit"}, P.data / "label_mapping.json")
    if P.splits.exists():
        splits = load_splits(P)
        D.check_splits(splits)
        return splits
    train = D.load_train(P.raw)
    save_json(D.audit_train(train, cfg["split"]["near_duplicate_threshold"]), P.data / "train_audit.json")
    split_cfg = D.SplitConfig(seed=cfg["seed"], n_folds=cfg["split"]["n_folds"],
                              model_val_fold=cfg["split"]["model_val_fold"],
                              routing_val_fold=cfg["split"]["routing_val_fold"])
    splits = D.make_splits(train, split_cfg)
    cols = ["row_id", "source_split", "partition", "group_id", "label_id", "label_name",
            "normalized_text_hash", "raw_text"]
    P.data.mkdir(parents=True, exist_ok=True)
    splits[cols].to_csv(P.splits, index=False, lineterminator="\n")
    summary = D.split_summary(splits)
    summary["splits_csv_sha256"] = sha256_file(P.splits)
    summary["config_hash"] = cfg["_hash"]
    save_json(summary, P.data / "split_summary.json")
    return splits[cols]


def load_splits(P: Paths) -> pd.DataFrame:
    df = pd.read_csv(P.splits, dtype={"raw_text": str}, keep_default_na=False)
    summary = P.data / "split_summary.json"
    if summary.exists():
        expected = load_json(summary).get("splits_csv_sha256")
        if expected and expected != sha256_file(P.splits):
            raise ValueError("splits.csv changed after it was frozen")
    return df


def preprocessing_examples(splits: pd.DataFrame, labels: list[str], n: int = 12, seed: int = 42) -> pd.DataFrame:
    """Fitting-partition examples shown under every lexical variant + encoder input."""
    fit = splits[splits.partition == "fit"]
    pattern = r"n't|n’t|\bcant\b|\d|[£$€]|top-?up|\n"
    pick = pd.concat([fit[fit.raw_text.str.contains(pattern, case=False, regex=True)].sample(n - 4, random_state=seed),
                      fit.sample(4, random_state=seed)])
    prot = protected_words(labels)
    out = {"row_id": pick.row_id.tolist(), "label_name": pick.label_name.tolist(),
           "raw_text": pick.raw_text.tolist(), "encoder_input": encoder_input(pick.raw_text.tolist())}
    for name, vcfg in LEXICAL_VARIANTS.items():
        if name == "L0_sklearn_default":
            from sklearn.feature_extraction.text import TfidfVectorizer

            an = TfidfVectorizer().build_analyzer()
            out[name] = [" ".join(an(t)) for t in pick.raw_text]
        else:
            out[name] = preprocess_lexical(pick.raw_text.tolist(), vcfg, prot)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# Shared grid machinery
# --------------------------------------------------------------------------------------

def _rounded(x: float) -> float:
    return round(float(x), 4)


def _pick_winner(rows: pd.DataFrame) -> pd.Series:
    r = rows.copy()
    r["_f1r"] = r["macro_f1"].map(_rounded)
    r["_nfeat"] = r["n_features"].fillna(0)
    r = r.sort_values(["_f1r", "_nfeat", "C", "grid_order"], ascending=[False, True, True, True])
    return r.iloc[0]


def _save_val_predictions(P: Paths, family: str, val: pd.DataFrame, scores: np.ndarray,
                          score_type: str, labels: list[str], extra: dict) -> dict:
    order = np.argsort(-scores, axis=1, kind="stable")
    pred = order[:, 0]
    top1 = scores[np.arange(len(pred)), order[:, 0]]
    top2 = scores[np.arange(len(pred)), order[:, 1]]
    df = pd.DataFrame({
        "row_id": val.row_id.values, "label_id": val.label_id.values, "label_name": val.label_name.values,
        "pred_id": pred, "pred_name": [labels[i] for i in pred],
        "top3_ids": [" ".join(map(str, o[:3])) for o in order],
        "score_type": score_type, "top1_score": top1, "margin_top1_top2": top1 - top2,
    })
    P.pred_val.mkdir(parents=True, exist_ok=True)
    csv_path = P.pred_val / f"{family}.csv"
    df.to_csv(csv_path, index=False, lineterminator="\n")
    np.save(P.pred_val / f"{family}_scores.npy", scores.astype(np.float32))
    meta = {"family": family, "class_order": "label_id 0..76 (label_mapping.json)",
            "row_order_checksum": row_order_checksum(val.row_id), "score_type": score_type,
            "predictions_sha256": sha256_file(csv_path), **extra}
    save_json(meta, P.pred_val / f"{family}_meta.json")
    return meta


# --------------------------------------------------------------------------------------
# Stage: lexical (C1, C2 and the preprocessing ablation)
# --------------------------------------------------------------------------------------

def lexical_reps(cfg: dict) -> list[LexicalRep]:
    L = cfg["lexical"]
    reps = [LexicalRep("word", v, tuple(ng)) for v in L["word_variants"] for ng in L["word_ngrams"]]
    for kind in L["char_kinds"]:
        reps.append(LexicalRep(kind, L["char_variant"], tuple(L["union_word_ngram"]), tuple(L["char_ngram"])))
    return reps


def run_lexical(root: str | Path = ".", verbose: bool = True) -> pd.DataFrame:
    P = Paths(root)
    cfg = load_config(P)
    splits = load_splits(P)
    labels = D.load_label_names(P.raw)
    prot = protected_words(labels)
    fit = splits[splits.partition == "fit"].sort_values("row_id")
    val = splits[splits.partition == "model_val"].sort_values("row_id")
    y_fit, y_val = fit.label_id.to_numpy(), val.label_id.to_numpy()

    # Stateless preprocessing, computed once per variant (nothing is fitted here).
    cache = {}
    for v in sorted({r.variant for r in lexical_reps(cfg)}):
        cache[v] = (preprocess_lexical(fit.raw_text.tolist(), LEXICAL_VARIANTS[v], prot),
                    preprocess_lexical(val.raw_text.tolist(), LEXICAL_VARIANTS[v], prot))

    rows, preds, grid_order = [], {}, 0
    L = cfg["lexical"]

    def run_one(rep: LexicalRep, clf: str, C: float, extension: bool = False):
        nonlocal grid_order
        Xf, Xv = cache[rep.variant]
        fr = fit_with_diagnostics(make_lexical_model(rep, clf, C), Xf, y_fit)
        scores, _ = class_scores(fr.model, Xv)
        pred = scores.argmax(1)
        m = classification_metrics(y_val, pred, scores)
        cid = f"{clf}|{rep.rep_id}|C={C:g}"
        preds[cid] = pred
        row = {"config_id": cid, "family": "C1" if clf == "lr" else "C2", "classifier": clf,
               "rep_id": rep.rep_id, "kind": rep.kind, "variant": rep.variant,
               "word_ngram": "-".join(map(str, rep.word_ngram)) if rep.kind != "char" else "",
               "char_ngram": "-".join(map(str, rep.char_ngram)) if rep.kind != "word" else "",
               "C": C, "edge_extension": extension, "n_features": n_features(fr.model),
               "fit_seconds": round(fr.fit_seconds, 3), "converged": fr.converged, "n_iter": fr.n_iter,
               "grid_order": grid_order, **m}
        grid_order += 1
        rows.append(row)
        if verbose:
            print(f"{cid:55s} macroF1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
                  f"fit={fr.fit_seconds:.1f}s conv={fr.converged}", flush=True)
        return fr

    for rep in lexical_reps(cfg):
        for clf in L["classifiers"]:
            for C in L["C_grid"]:
                run_one(rep, clf, C)
    grid = pd.DataFrame(rows)

    # Within-family winners + pre-registered edge rule.
    winners = {}
    for fam, clf in (("C1", "lr"), ("C2", "svm")):
        w = _pick_winner(grid[grid.classifier == clf])
        edge = None
        if w.C == min(L["C_grid"]):
            edge = L["edge_extension"]["below"]
        elif w.C == max(L["C_grid"]):
            edge = L["edge_extension"]["above"]
        if edge is not None:
            rep = next(r for r in lexical_reps(cfg) if r.rep_id == w.rep_id)
            run_one(rep, clf, edge, extension=True)
            ext = pd.DataFrame(rows[-1:])
            if _rounded(ext.macro_f1.iloc[0]) > _rounded(w.macro_f1):
                w = ext.iloc[0]
        winners[fam] = w
    grid = pd.DataFrame(rows)
    P.dev.mkdir(parents=True, exist_ok=True)
    grid.to_csv(P.dev / "lexical_grid.csv", index=False, lineterminator="\n")
    np.savez_compressed(P.dev / "lexical_val_predictions.npz",
                        row_ids=val.row_id.to_numpy(), y_true=y_val,
                        **{k.replace("|", "__").replace(":", "~").replace("=", "-"): v for k, v in preds.items()})

    # Refit winners (deterministic) to persist them as raw-text pipelines.
    idx = bootstrap_indices(len(y_val), cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
    candidates = {}
    for fam, w in winners.items():
        rep = next(r for r in lexical_reps(cfg) if r.rep_id == w.rep_id)
        Xf, Xv = cache[rep.variant]
        fr = fit_with_diagnostics(make_lexical_model(rep, w.classifier, float(w.C)), Xf, y_fit)
        scores, stype = class_scores(fr.model, Xv)
        if not np.array_equal(scores.argmax(1), preds[w.config_id]):
            raise AssertionError("Refit did not reproduce grid predictions")
        pipe = deployable_lexical_pipeline(fr.model, rep, labels)
        raw_scores, _ = class_scores(pipe, val.raw_text.tolist())
        if not np.allclose(raw_scores, scores):
            raise AssertionError("Raw-text pipeline does not reproduce cached-string scores")
        P.models.mkdir(parents=True, exist_ok=True)
        model_path = P.models / f"{fam}.joblib"
        joblib.dump(pipe, model_path)
        lat = latency_batch_one(lambda xs: pipe.predict(xs), val.raw_text.tolist(),
                                cfg["latency"]["warmup"], cfg["latency"]["n"])
        cand = {
            "family": fam, "config_id": w.config_id, "classifier": w.classifier, "C": float(w.C),
            "representation": rep.to_dict(), "preprocessing": LEXICAL_VARIANTS[rep.variant].to_dict(),
            "score_type": stype, "n_fit_rows": int(len(y_fit)), "n_features": int(w.n_features),
            "val_metrics": classification_metrics(y_val, scores.argmax(1), scores),
            "val_ci": bootstrap_ci(y_val, scores.argmax(1), idx),
            "fit_seconds": float(w.fit_seconds), "converged": bool(w.converged),
            "artifact": str(model_path.relative_to(P.root)), "artifact_sha256": sha256_file(model_path),
            "artifact_bytes": model_path.stat().st_size,
            "latency_this_machine": lat, "environment": environment_info(),
            "config_hash": cfg["_hash"],
        }
        cand["predictions"] = _save_val_predictions(P, fam, val, scores, stype, labels,
                                                    {"config_id": w.config_id})
        save_json(cand, P.dev / f"candidate_{fam}.json")
        candidates[fam] = cand

    ablation = lexical_ablation(grid, preds, y_val, idx)
    ablation.to_csv(P.dev / "lexical_ablation.csv", index=False, lineterminator="\n")
    preprocessing_examples(splits, labels).to_csv(P.dev / "preprocessing_examples.csv", index=False,
                                                   lineterminator="\n")
    return grid


ABLATION_ORDER = [
    ("word", "L0_sklearn_default", None),
    ("word", "L1_normalize_tokenize", "L0_sklearn_default"),
    ("word", "L2_contractions", "L1_normalize_tokenize"),
    ("word", "L3_numbers", "L2_contractions"),
    ("word", "L4_lemma", "L3_numbers"),
    ("word", "L5_stop_protected", "L3_numbers"),
    ("word", "L6_stop_sklearn", "L3_numbers"),
    ("char", CHAR_BASE_VARIANT, "L3_numbers"),
    ("word+char", CHAR_BASE_VARIANT, "L3_numbers"),
]


def lexical_ablation(grid: pd.DataFrame, preds: dict, y_val: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    """Best tuned config per (representation step, classifier), with paired deltas.

    'previous' is the step the row adds to (cumulative chain L0->L3; L4-L6 and the
    character variants are each compared with L3 word features)."""
    out = []
    for clf in ("lr", "svm"):
        g = grid[(grid.classifier == clf) & (~grid.edge_extension)]
        best = {}
        for kind, variant, _ in ABLATION_ORDER:
            sub = g[(g.kind == kind) & (g.variant == variant)]
            best[(kind, variant)] = _pick_winner(sub)
        ref = best[("word", "L0_sklearn_default")]
        for kind, variant, prev in ABLATION_ORDER:
            w = best[(kind, variant)]
            row = {"classifier": clf, "step": f"{kind}:{variant}", "best_config": w.config_id,
                   "word_ngram": w.word_ngram, "C": w.C, "n_features": w.n_features,
                   "macro_f1": w.macro_f1, "accuracy": w.accuracy, "top3_accuracy": w.top3_accuracy,
                   **bootstrap_ci(y_val, preds[w.config_id], idx)}
            d0 = paired_bootstrap_delta(y_val, preds[ref.config_id], preds[w.config_id], idx)
            row.update({"delta_vs_L0": d0["delta_macro_f1"], "delta_vs_L0_ci": d0["delta_ci"]})
            if prev is not None:
                pw = best[("word", prev)]
                dp = paired_bootstrap_delta(y_val, preds[pw.config_id], preds[w.config_id], idx)
                row.update({"compared_with": f"word:{prev}", "delta_vs_prev": dp["delta_macro_f1"],
                            "delta_vs_prev_ci": dp["delta_ci"], "p_delta_le_0": dp["p_delta_le_0"],
                            "n_disagreements": dp["n_disagreements"]})
            out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# Stage: embeddings (C3, C4)
# --------------------------------------------------------------------------------------

def embed_partitions(P: Paths, cfg: dict, splits: pd.DataFrame, encoder=None) -> dict:
    """Encode fit + model_val + routing_val once and cache with row-order checksums.

    Only official-training rows are encoded here. Test inputs are encoded in the
    final stage. Held-out vectors are never used as training examples."""
    E = cfg["embedding"]
    dev = splits.sort_values("row_id")
    cache_npy = P.emb / "minilm_dev.npy"
    meta_path = P.emb / "minilm_dev_meta.json"
    if cache_npy.exists() and meta_path.exists():
        meta = load_json(meta_path)
        if meta["row_order_checksum"] != row_order_checksum(dev.row_id):
            raise ValueError("Cached embeddings do not match the frozen row order")
        if meta["encoder"]["model_name"] != E["model_name"] or (
                E["revision"] and meta["encoder"]["resolved_revision"] not in (E["revision"], None)):
            raise ValueError("Cached embeddings were produced by a different encoder/revision")
        return {"X": np.load(cache_npy), "row_ids": dev.row_id.to_numpy(), "meta": meta}
    if encoder is None:
        from .models import SentenceEncoder

        encoder = SentenceEncoder(E["model_name"], E["revision"], batch_size=E["batch_size"])
    texts = encoder_input(dev.raw_text.tolist())
    import time

    t0 = time.perf_counter()
    X = encoder.encode(texts)
    secs = time.perf_counter() - t0
    lengths = encoder.token_lengths(texts) if hasattr(encoder, "token_lengths") else None
    info = dict(encoder.info)
    meta = {"encoder": info, "rows": int(len(dev)), "row_order_checksum": row_order_checksum(dev.row_id),
            "encode_seconds": round(secs, 2), "input": E["input"],
            "token_length": None if lengths is None else {
                "max": int(lengths.max()), "p95": float(np.percentile(lengths, 95)),
                "over_max_seq_length": int((lengths > info.get("max_seq_length", 10**9)).sum())},
            "norms": {"min": float(np.linalg.norm(X, axis=1).min()), "max": float(np.linalg.norm(X, axis=1).max())},
            "environment": environment_info()}
    P.emb.mkdir(parents=True, exist_ok=True)
    np.save(cache_npy, X)
    meta["npy_sha256"] = sha256_file(cache_npy)
    save_json(meta, meta_path)
    return {"X": X, "row_ids": dev.row_id.to_numpy(), "meta": meta}


def run_embeddings(root: str | Path = ".", encoder=None, verbose: bool = True) -> pd.DataFrame:
    P = Paths(root)
    cfg = load_config(P)
    E = cfg["embedding"]
    splits = load_splits(P)
    labels = D.load_label_names(P.raw)
    emb = embed_partitions(P, cfg, splits, encoder)
    pos = {r: i for i, r in enumerate(emb["row_ids"])}
    fit = splits[splits.partition == "fit"].sort_values("row_id")
    val = splits[splits.partition == "model_val"].sort_values("row_id")
    Xf, Xv = emb["X"][[pos[r] for r in fit.row_id]], emb["X"][[pos[r] for r in val.row_id]]
    y_fit, y_val = fit.label_id.to_numpy(), val.label_id.to_numpy()

    rows, preds, models, grid_order = [], {}, {}, 0

    def run_one(clf, C, extension=False):
        nonlocal grid_order
        fr = fit_with_diagnostics(make_embedding_model(clf, C), Xf, y_fit)
        scores, _ = class_scores(fr.model, Xv)
        m = classification_metrics(y_val, scores.argmax(1), scores)
        cid = f"{clf}|minilm|C={C:g}"
        preds[cid], models[cid] = scores.argmax(1), fr.model
        rows.append({"config_id": cid, "family": "C3" if clf == "lr" else "C4", "classifier": clf,
                     "C": C, "edge_extension": extension, "n_features": Xf.shape[1],
                     "fit_seconds": round(fr.fit_seconds, 3), "converged": fr.converged,
                     "n_iter": fr.n_iter, "grid_order": grid_order, **m})
        grid_order += 1
        if verbose:
            print(f"{cid:30s} macroF1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}", flush=True)

    for clf in E["classifiers"]:
        for C in E["C_grid"]:
            run_one(clf, C)
    grid = pd.DataFrame(rows)
    idx = bootstrap_indices(len(y_val), cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
    for fam, clf in (("C3", "lr"), ("C4", "svm")):
        w = _pick_winner(grid[grid.classifier == clf])
        edge = (E["edge_extension"]["below"] if w.C == min(E["C_grid"]) else
                E["edge_extension"]["above"] if w.C == max(E["C_grid"]) else None)
        if edge is not None:
            run_one(clf, edge, extension=True)
            ext = pd.DataFrame(rows[-1:]).iloc[0]
            if _rounded(ext.macro_f1) > _rounded(w.macro_f1):
                w = ext
        model = models[w.config_id]
        scores, stype = class_scores(model, Xv)
        rev = emb["meta"]["encoder"].get("resolved_revision") or E["revision"]
        pipe = deployable_embedding_pipeline(model, E["model_name"], rev)
        P.models.mkdir(parents=True, exist_ok=True)
        model_path = P.models / f"{fam}.joblib"
        joblib.dump(pipe, model_path)
        cand = {"family": fam, "config_id": w.config_id, "classifier": clf, "C": float(w.C),
                "representation": {"kind": "minilm", "encoder": emb["meta"]["encoder"]},
                "preprocessing": {"encoder_input": E["input"]}, "score_type": stype,
                "n_fit_rows": int(len(y_fit)), "n_features": int(Xf.shape[1]),
                "val_metrics": classification_metrics(y_val, scores.argmax(1), scores),
                "val_ci": bootstrap_ci(y_val, scores.argmax(1), idx),
                "fit_seconds": float(w.fit_seconds), "converged": bool(w.converged),
                "embedding_cache_sha256": emb["meta"].get("npy_sha256"),
                "artifact": str(model_path.relative_to(P.root)), "artifact_sha256": sha256_file(model_path),
                "artifact_bytes": model_path.stat().st_size, "environment": environment_info(),
                "config_hash": cfg["_hash"]}
        cand["predictions"] = _save_val_predictions(P, fam, val, scores, stype, labels, {"config_id": w.config_id})
        save_json(cand, P.dev / f"candidate_{fam}.json")
    grid = pd.DataFrame(rows)
    P.dev.mkdir(parents=True, exist_ok=True)
    grid.to_csv(P.dev / "embedding_grid.csv", index=False, lineterminator="\n")
    return grid


# --------------------------------------------------------------------------------------
# Stage: select (freeze)
# --------------------------------------------------------------------------------------

def select(root: str | Path = ".", measure_latency: bool = True) -> dict:
    """Deployment choice among C1-C4 under the plan's rule; writes selection.json.

    Latency for every candidate is (re)measured here, on this machine, from the
    saved raw-text pipelines, so the tie-break compares like with like."""
    P = Paths(root)
    cfg = load_config(P)
    missing = [f for f in FAMILIES if not (P.dev / f"candidate_{f}.json").exists()]
    if missing:
        raise FileNotFoundError(f"Missing candidates {missing}: run the 'lexical' and 'embeddings' stages first")
    splits = load_splits(P)
    val = splits[splits.partition == "model_val"].sort_values("row_id")
    cands = {f: load_json(P.dev / f"candidate_{f}.json") for f in FAMILIES}
    for f, c in cands.items():
        if sha256_file(P.root / c["artifact"]) != c["artifact_sha256"]:
            raise ValueError(f"{f} artifact changed after it was recorded")
        if measure_latency:
            pipe = joblib.load(P.root / c["artifact"])
            c["latency_selection_machine"] = latency_batch_one(
                lambda xs: pipe.predict(xs), val.raw_text.tolist(), cfg["latency"]["warmup"], cfg["latency"]["n"])
    order = list(FAMILIES)
    ranked = sorted(order, key=lambda f: (-_rounded(cands[f]["val_metrics"]["macro_f1"]),
                                          cands[f].get("latency_selection_machine", {}).get("p50_ms", 0),
                                          order.index(f)))
    selection = {"deployment_family": ranked[0], "ranking": ranked,
                 "rule": cfg["deployment_selection"], "candidates": cands,
                 "splits_csv_sha256": sha256_file(P.splits), "config_hash": cfg["_hash"],
                 "environment": environment_info(),
                 "note": "Frozen. Test predictions for all four families are produced once by the 'final' stage."}
    save_json(selection, P.frozen / "selection.json")
    return selection


# --------------------------------------------------------------------------------------
# Stage: final (gated)
# --------------------------------------------------------------------------------------

def check_frozen(P: "Paths") -> dict:
    """Both freezes must exist before any test input is read, and routing.json must
    have been made from the current selection.json."""
    sel_path, routing_path = P.frozen / "selection.json", P.frozen / "routing.json"
    if not sel_path.exists():
        raise D.TestSetLockedError("No frozen selection.json; run the 'select' stage first")
    if not routing_path.exists():
        raise D.TestSetLockedError("No frozen routing.json; run the 'routing' stage first")
    routing = load_json(routing_path)
    if routing.get("selection_sha256") != sha256_file(sel_path):
        raise D.TestSetLockedError("routing.json was not made from the current selection.json; "
                                   "re-run the 'routing' stage")
    return load_json(sel_path)


def predict_test(root: str | Path = ".", *, i_confirm_configuration_is_frozen: bool = False) -> dict:
    """Write test predictions for every frozen family. Reads no test labels.

    All four families are predicted in memory first; files are written only if every
    family succeeded, so a crash leaves no partial predictions. Any exception (for
    example an encoder that cannot be loaded) propagates instead of being recorded
    as 3,080 invalid predictions."""
    P = Paths(root)
    sel = check_frozen(P)
    labels = D.load_label_names(P.raw)
    for fam in sel["candidates"]:
        if (P.pred_test / f"{fam}.csv").exists():
            raise FileExistsError(f"Test predictions for {fam} already exist; the final pass runs once")
    test = D.load_test_texts(P.raw, i_confirm_configuration_is_frozen=i_confirm_configuration_is_frozen)
    outputs = {}
    for fam, c in sel["candidates"].items():
        if sha256_file(P.root / c["artifact"]) != c["artifact_sha256"]:
            raise ValueError(f"{fam} artifact differs from the frozen record")
        pipe = joblib.load(P.root / c["artifact"])
        scores, stype = class_scores(pipe, test.raw_text.tolist())
        order = np.argsort(-scores, axis=1, kind="stable")
        pred = order[:, 0]
        df = pd.DataFrame({"row_id": test.row_id, "pred_id": pred,
                           "pred_name": [labels[i] for i in pred],
                           "top3_ids": [" ".join(map(str, o[:3])) for o in order], "score_type": stype})
        outputs[fam] = (df, scores)
    P.pred_test.mkdir(parents=True, exist_ok=True)
    written = {}
    for fam, (df, scores) in outputs.items():
        df.to_csv(P.pred_test / f"{fam}.csv", index=False, lineterminator="\n")
        np.save(P.pred_test / f"{fam}_scores.npy", scores.astype(np.float32))
        written[fam] = sha256_file(P.pred_test / f"{fam}.csv")
    save_json({"predictions_sha256": written, "row_order_checksum": row_order_checksum(test.row_id),
               "environment": environment_info()}, P.pred_test / "manifest.json")
    return written


def score_test(root: str | Path = ".", *, i_confirm_configuration_is_frozen: bool = False) -> dict:
    """Separate scoring function: reads persisted predictions, then the answers."""
    P = Paths(root)
    manifest = load_json(P.pred_test / "manifest.json")
    labels = D.load_label_names(P.raw)
    answers = D.load_test_labels(P.raw, i_confirm_configuration_is_frozen=i_confirm_configuration_is_frozen)
    test_hash = D.load_test_texts(P.raw, i_confirm_configuration_is_frozen=i_confirm_configuration_is_frozen)[
        ["row_id", "normalized_text_hash"]]
    train = load_splits(P)
    overlap = test_hash.normalized_text_hash.isin(set(train.normalized_text_hash)).to_numpy()
    test_texts = D.load_test_texts(P.raw, i_confirm_configuration_is_frozen=i_confirm_configuration_is_frozen)
    train_loose = set(train.raw_text.map(D.loose_duplicate_key))
    loose = test_texts.set_index("row_id").loc[answers.row_id, "raw_text"].map(D.loose_duplicate_key) \
        .isin(train_loose).to_numpy()
    results = {"n_test": int(len(answers)), "exact_overlap_with_train": int(overlap.sum()),
               "loose_overlap_with_train": int(loose.sum()),
               "overlap_note": "exact = NFKC + casefold + whitespace key; loose = also ignores punctuation "
                               "(differences such as '?', hyphens, currency symbols)",
               "families": {}}
    preds = {}
    for fam, sha in manifest["predictions_sha256"].items():
        path = P.pred_test / f"{fam}.csv"
        if sha256_file(path) != sha:
            raise ValueError(f"{fam} predictions changed after they were written")
        pred = pd.read_csv(path).set_index("row_id").loc[answers.row_id, "pred_id"].to_numpy()
        scores_path = P.pred_test / f"{fam}_scores.npy"
        scores = np.load(scores_path) if scores_path.exists() else None
        y = answers.label_id.to_numpy()
        res = {"all": classification_metrics(y, pred, scores),
               "overlap_excluded_sensitivity": classification_metrics(
                   y[~overlap], pred[~overlap], None if scores is None else scores[~overlap]),
               "loose_overlap_excluded_sensitivity": classification_metrics(
                   y[~loose], pred[~loose], None if scores is None else scores[~loose])}
        P.final.mkdir(parents=True, exist_ok=True)
        per_class_table(y, pred, labels).to_csv(P.final / f"{fam}_per_class.csv", index=False, lineterminator="\n")
        top_confusions(y, pred, labels, 20).to_csv(P.final / f"{fam}_top_confusions.csv", index=False,
                                                   lineterminator="\n")
        results["families"][fam] = res
        preds[fam] = pred
    save_json(results, P.final / "test_metrics.json")
    save_json(bootstrap_test_uncertainty(answers.label_id.to_numpy(), preds, load_config(P)["bootstrap"]),
              P.final / "test_ci_and_paired.json")
    return results


def bootstrap_test_uncertainty(y: np.ndarray, preds: dict, boot: dict) -> dict:
    """Percentile bootstrap CIs per family and paired differences for every pair,
    on one shared set of resamples of the test requests. Key "Cj-Ci" = Cj minus Ci."""
    idx = bootstrap_indices(len(y), boot["n_boot"], boot["seed"])
    fams = sorted(preds)
    return {
        "bootstrap": {**boot, "unit": "test request", "macro_f1": "averaged over intents present in each resample"},
        "test_ci": {f: bootstrap_ci(y, preds[f], idx) for f in fams},
        "test_paired": {f"{b}-{a}": paired_bootstrap_delta(y, preds[a], preds[b], idx)
                        for i, a in enumerate(fams) for b in fams[i + 1:]},
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "lexical", "embeddings", "select", "routing", "final"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--i-confirm-freeze", action="store_true",
                    help="required for 'final': confirms selection.json and routing.json are frozen")
    a = ap.parse_args(argv)
    if a.stage == "prepare":
        prepare(a.root)
    elif a.stage == "lexical":
        run_lexical(a.root)
    elif a.stage == "embeddings":
        run_embeddings(a.root)
    elif a.stage == "select":
        print(json.dumps({k: v for k, v in select(a.root).items() if k != "candidates"}, indent=2))
    elif a.stage == "routing":
        from .routing import freeze_routing

        print(json.dumps(freeze_routing(a.root), indent=2))
    elif a.stage == "final":
        from .routing import score_routing_and_slice

        predict_test(a.root, i_confirm_configuration_is_frozen=a.i_confirm_freeze)
        print(json.dumps(score_test(a.root, i_confirm_configuration_is_frozen=a.i_confirm_freeze), indent=2))
        print(json.dumps(score_routing_and_slice(a.root, i_confirm_configuration_is_frozen=a.i_confirm_freeze),
                         indent=2))


if __name__ == "__main__":
    main()
