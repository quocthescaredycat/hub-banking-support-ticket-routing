"""Routing threshold (PLAN §6) and the lexical-difficulty test slice (PLAN §7).

Routing: for the frozen deployment classifier, the uncertainty signal is the
top-1 minus top-2 score margin. On ``routing_val`` (never used for fitting or
model selection) choose the threshold with the greatest coverage such that the
observed intent error rate among automatically routed requests is at most 5% and
at least 100 requests are routed; ties prefer the larger threshold. The threshold
is written to ``artifacts/frozen/routing.json`` before the test set is opened.

Difficulty slice: an auxiliary TF-IDF (whitespace tokens, punctuation ignored,
independent of the C1/C2 preprocessing) is fitted on fitting-partition texts; one
concatenated profile per intent. After test predictions are locked, the five test
requests per true intent least similar to their intent's profile form a 385-row
slice (inspired by Qi et al., 2021; not an exact reproduction).
"""

from __future__ import annotations

import re

import joblib
import numpy as np
import pandas as pd

from . import data as D
from .artifacts import load_json, save_json, sha256_file
from .evaluation import classification_metrics
from .models import class_scores

MAX_ERROR = 0.05
MIN_ROUTED = 100


def choose_threshold(margins: np.ndarray, correct: np.ndarray,
                     max_error: float = MAX_ERROR, min_routed: int = MIN_ROUTED) -> dict:
    """Greatest coverage s.t. error among routed <= max_error and routed >= min_routed.
    Requests with margin >= t are routed. Ties (same coverage) prefer larger t."""
    candidates = np.unique(margins)
    best = None
    for t in candidates:
        routed = margins >= t
        n = int(routed.sum())
        if n < min_routed:
            continue
        err = float(1 - correct[routed].mean())
        if err > max_error:
            continue
        key = (n, t)                     # coverage first, then larger threshold
        if best is None or key > best[0]:
            best = (key, {"threshold": float(t), "n_routed": n, "coverage": n / len(margins),
                          "error_rate_routed": err})
    if best is None:
        return {"threshold": None, "n_routed": 0, "coverage": 0.0, "error_rate_routed": None,
                "note": "no threshold satisfies the rule; route everything to review"}
    return best[1]


def margins_of(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="stable")
    r = np.arange(len(scores))
    return order[:, 0], scores[r, order[:, 0]] - scores[r, order[:, 1]]


def freeze_routing(root=".") -> dict:
    from .experiments import Paths, load_splits

    P = Paths(root)
    sel = load_json(P.frozen / "selection.json")
    fam = sel["deployment_family"]
    c = sel["candidates"][fam]
    if sha256_file(P.root / c["artifact"]) != c["artifact_sha256"]:
        raise ValueError("deployment artifact differs from the frozen record")
    pipe = joblib.load(P.root / c["artifact"])
    rv = load_splits(P).query("partition == 'routing_val'").sort_values("row_id")
    scores, stype = class_scores(pipe, rv.raw_text.tolist())
    pred, margin = margins_of(scores)
    correct = pred == rv.label_id.to_numpy()
    res = choose_threshold(margin, correct)
    out = {"deployment_family": fam, "score_type": stype, "signal": "top1 - top2 score margin",
           "rule": f"max coverage s.t. error among routed <= {MAX_ERROR:.0%} and routed >= {MIN_ROUTED}; "
                   "ties prefer the larger threshold",
           "routing_val_rows": int(len(rv)), "routing_val_accuracy": float(correct.mean()), **res,
           "selection_sha256": sha256_file(P.frozen / "selection.json")}
    save_json(out, P.frozen / "routing.json")
    return out


# --------------------------------------------------------------------------------------
# Lexical-difficulty slice
# --------------------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]")


def _ws_tokens(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def difficulty_slice(fit_texts, fit_labels, test_texts, test_labels, per_intent: int = 5) -> np.ndarray:
    """Positions of the ``per_intent`` test rows per true intent least similar to
    that intent's concatenated fitting profile (ties by position)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(tokenizer=_ws_tokens, token_pattern=None, lowercase=False)
    vec.fit(list(fit_texts))
    fit_labels, test_labels = np.asarray(fit_labels), np.asarray(test_labels)
    labels = np.unique(fit_labels)
    profiles = vec.transform([" ".join(np.asarray(fit_texts)[fit_labels == k]) for k in labels])
    Xt = vec.transform(list(test_texts))
    prof_row = {k: i for i, k in enumerate(labels)}
    sims = np.asarray([Xt[i].multiply(profiles[prof_row[k]]).sum() for i, k in enumerate(test_labels)])
    keep = []
    for k in labels:
        pos = np.where(test_labels == k)[0]
        keep.extend(pos[np.lexsort((pos, sims[pos]))][:per_intent])
    return np.sort(np.array(keep))


def score_routing_and_slice(root=".", *, i_confirm_configuration_is_frozen: bool = False) -> dict:
    """After the final stage: routed coverage/error on test, and the difficulty slice
    for every family, from persisted predictions only."""
    from .experiments import Paths, load_splits

    P = Paths(root)
    routing = load_json(P.frozen / "routing.json")
    if routing.get("selection_sha256") != sha256_file(P.frozen / "selection.json"):
        raise ValueError("routing.json was not made from the current selection.json")
    answers = D.load_test_labels(P.raw, i_confirm_configuration_is_frozen=i_confirm_configuration_is_frozen)
    texts = D.load_test_texts(P.raw, i_confirm_configuration_is_frozen=i_confirm_configuration_is_frozen)
    y = answers.label_id.to_numpy()
    fit = load_splits(P).query("partition == 'fit'").sort_values("row_id")
    sl = difficulty_slice(fit.raw_text, fit.label_id, texts.raw_text, y)
    out = {"slice": {"n": int(len(sl)), "rule": "5 least similar test rows per true intent to the intent's "
                     "fitting-partition TF-IDF profile (whitespace tokens, punctuation ignored)"},
           "families": {}}
    for fam in ("C1", "C2", "C3", "C4"):
        path = P.pred_test / f"{fam}.csv"
        if not path.exists():
            continue
        pred = pd.read_csv(path).set_index("row_id").loc[answers.row_id, "pred_id"].to_numpy()
        sp = P.pred_test / f"{fam}_scores.npy"
        scores = np.load(sp) if sp.exists() else None
        out["families"][fam] = {"difficulty_slice": classification_metrics(
            y[sl], pred[sl], None if scores is None else scores[sl])}
        if fam == routing["deployment_family"] and routing["threshold"] is None:
            out["routing_test"] = {"family": fam, "threshold": None, "n_routed": 0, "coverage": 0.0,
                                   "error_rate_routed": None, "n_review": int(len(y))}
        elif fam == routing["deployment_family"] and scores is not None:
            _, margin = margins_of(scores)
            routed = margin >= routing["threshold"]
            out["routing_test"] = {"family": fam, "threshold": routing["threshold"],
                                   "n_routed": int(routed.sum()), "coverage": float(routed.mean()),
                                   "error_rate_routed": float(1 - (pred[routed] == y[routed]).mean()),
                                   "n_review": int((~routed).sum())}
    save_json(out, P.final / "routing_and_slice.json")
    pd.DataFrame({"row_id": answers.row_id.to_numpy()[sl]}).to_csv(P.final / "difficulty_slice_rows.csv",
                                                                    index=False, lineterminator="\n")
    return out
