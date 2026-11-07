"""Lexical (TF-IDF) and frozen sentence-embedding classifiers.

Representations
  * word TF-IDF on a preprocessing variant from ``preprocessing.LEXICAL_VARIANTS``
  * character n-gram TF-IDF (``char_wb``) on the L3 string
  * word + character TF-IDF (feature union)
  * frozen ``sentence-transformers/all-MiniLM-L6-v2`` vectors, L2-normalised

Classifiers
  * Logistic Regression (multinomial, L2, lbfgs) -> ``predict_proba`` scores,
    labelled "probability (uncalibrated)"
  * Linear SVM (``LinearSVC``, one-vs-rest, squared hinge) -> ``decision_function``
    margins, labelled "decision score"; never shown as a percentage

The TF-IDF vocabulary and IDF are fitted inside the pipeline, so calling ``fit`` on
the fitting partition is the only place they are learned.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from .preprocessing import LEXICAL_VARIANTS, LexicalPreprocessor, protected_words

ClassifierName = Literal["lr", "svm"]

SCORE_TYPE = {
    "lr": "probability (multinomial LR, uncalibrated)",
    "svm": "decision score (LinearSVC margin, not a probability)",
}


def make_classifier(name: ClassifierName, C: float, seed: int = 42):
    if name == "lr":
        return LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    if name == "svm":
        return LinearSVC(C=C, max_iter=20000, random_state=seed)
    raise ValueError(name)


# --------------------------------------------------------------------------------------
# Lexical representation specs
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LexicalRep:
    kind: Literal["word", "char", "word+char"]
    variant: str                       # key in LEXICAL_VARIANTS
    word_ngram: tuple[int, int] = (1, 1)
    char_ngram: tuple[int, int] = (2, 5)

    @property
    def rep_id(self) -> str:
        if self.kind == "word":
            return f"word{self.word_ngram[0]}{self.word_ngram[1]}:{self.variant}"
        if self.kind == "char":
            return f"charwb{self.char_ngram[0]}{self.char_ngram[1]}:{self.variant}"
        return (f"word{self.word_ngram[0]}{self.word_ngram[1]}+charwb"
                f"{self.char_ngram[0]}{self.char_ngram[1]}:{self.variant}")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "variant": self.variant,
                "word_ngram": list(self.word_ngram), "char_ngram": list(self.char_ngram)}


def make_vectorizer(rep: LexicalRep):
    """TF-IDF featuriser that consumes the *preprocessed* strings."""
    if rep.variant == "L0_sklearn_default":
        if rep.kind != "word":
            raise ValueError("L0 is defined for word features only")
        # scikit-learn defaults on raw text: lowercase=True, token_pattern=r"(?u)\b\w\w+\b"
        return TfidfVectorizer(ngram_range=rep.word_ngram)
    word = TfidfVectorizer(ngram_range=rep.word_ngram, lowercase=False,
                           tokenizer=str.split, token_pattern=None)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=rep.char_ngram, lowercase=False)
    if rep.kind == "word":
        return word
    if rep.kind == "char":
        return char
    return FeatureUnion([("word", word), ("char", char)])


def make_lexical_model(rep: LexicalRep, clf: ClassifierName, C: float) -> Pipeline:
    """Model over preprocessed strings (used inside the grid, with cached strings)."""
    return Pipeline([("tfidf", make_vectorizer(rep)), ("clf", make_classifier(clf, C))])


def deployable_lexical_pipeline(fitted: Pipeline, rep: LexicalRep, label_names: Sequence[str]) -> Pipeline:
    """Prepend the stateless preprocessor so the saved artifact accepts raw text."""
    cfg = LEXICAL_VARIANTS[rep.variant].to_dict()
    prep = LexicalPreprocessor(config=cfg, protected=tuple(sorted(protected_words(label_names))))
    return Pipeline([("preprocess", prep)] + list(fitted.steps))


# --------------------------------------------------------------------------------------
# Fitting / scoring helpers shared by both families
# --------------------------------------------------------------------------------------

@dataclass
class FitResult:
    model: object
    fit_seconds: float
    converged: bool
    n_iter: int | None
    warnings: list[str] = field(default_factory=list)


def fit_with_diagnostics(model, X, y) -> FitResult:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - t0
    msgs = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    clf = model.steps[-1][1] if isinstance(model, Pipeline) else model
    n_iter = getattr(clf, "n_iter_", None)
    if n_iter is not None:
        n_iter = int(np.max(n_iter))
    return FitResult(model, elapsed, converged=not msgs, n_iter=n_iter, warnings=msgs[:3])


def class_scores(model, X) -> tuple[np.ndarray, str]:
    """[N, 77] score matrix in label-id order, plus its score type."""
    clf = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if not np.array_equal(clf.classes_, np.arange(len(clf.classes_))):
        raise AssertionError("classes_ must be 0..K-1 in label-id order")
    if isinstance(clf, LogisticRegression):
        return model.predict_proba(X), SCORE_TYPE["lr"]
    return model.decision_function(X), SCORE_TYPE["svm"]


def n_features(model) -> int | None:
    tf = model.steps[0][1] if isinstance(model, Pipeline) else None
    if isinstance(tf, TfidfVectorizer):
        return len(tf.vocabulary_)
    if isinstance(tf, FeatureUnion):
        return int(sum(len(t.vocabulary_) for _, t in tf.transformer_list))
    return None


# --------------------------------------------------------------------------------------
# Frozen sentence encoder (MiniLM)
# --------------------------------------------------------------------------------------

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def pooling_modes(cfg: dict) -> list[str]:
    """Pooling modes from a Pooling module config, in either format.

    sentence-transformers >= 6: ``{"pooling_mode": "mean"}`` or a list/tuple of modes.
    Older versions: one boolean key per mode, e.g. ``{"pooling_mode_mean_tokens": True}``."""
    mode = cfg.get("pooling_mode")
    if isinstance(mode, str):
        return [mode]
    if isinstance(mode, (list, tuple)):
        return [str(m) for m in mode]
    return [k.replace("pooling_mode_", "") for k, v in cfg.items() if k.startswith("pooling_mode_") and v]


class SentenceEncoder:
    """Thin wrapper around ``sentence_transformers.SentenceTransformer``.

    Records what the plan asks for: model name, resolved revision, embedding size,
    maximum sequence length, pooling and whether a Normalize module is present.
    ``revision=None`` uses the hub default and records the commit that was resolved;
    put that hash in ``configs/experiment.json`` afterwards to pin it.
    """

    def __init__(self, model_name: str = MINILM, revision: str | None = None,
                 device: str | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, revision=revision, device=device)
        self.info = self._describe(revision)

    def _describe(self, requested_revision):
        modules = [type(m).__name__ for m in self.model]
        pooling = None
        for m in self.model:
            if type(m).__name__ == "Pooling":
                pooling = pooling_modes(m.get_config_dict())
        resolved = None
        try:  # the snapshot folder name in the HF cache is the commit hash
            from pathlib import Path as _Path

            from huggingface_hub import snapshot_download

            path = snapshot_download(self.model_name, revision=requested_revision, local_files_only=True)
            resolved = _Path(path).name  # works for both / and \ separators
        except Exception:
            pass
        import sentence_transformers

        return {
            "model_name": self.model_name,
            "requested_revision": requested_revision,
            "resolved_revision": resolved,
            "embedding_dim": int(self._embedding_dimension()),
            "max_seq_length": int(self.model.max_seq_length),
            "modules": modules,
            "pooling": pooling,
            "normalize_module": "Normalize" in modules,
            "device": str(self.model.device),
            "sentence_transformers_version": sentence_transformers.__version__,
        }

    def _embedding_dimension(self) -> int:
        # renamed in sentence-transformers 6 (the old name still works there, with a warning)
        if hasattr(type(self.model), "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def token_lengths(self, texts: Sequence[str]) -> np.ndarray:
        tok = self.model.tokenizer
        return np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)


def make_embedding_model(clf: ClassifierName, C: float) -> Pipeline:
    """Classifier over cached embeddings. ``Normalizer`` is stateless (row-wise L2)."""
    return Pipeline([("l2", Normalizer()), ("clf", make_classifier(clf, C))])


class EncoderTransformer:
    """sklearn-style stateless step that turns raw text into MiniLM vectors, so a
    saved embedding pipeline accepts raw text. The encoder is not pickled; it is
    reloaded (same name and revision) on first use."""

    def __init__(self, model_name: str = MINILM, revision: str | None = None):
        self.model_name = model_name
        self.revision = revision
        self._enc = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from .preprocessing import encoder_input

        if self._enc is None:
            self._enc = SentenceEncoder(self.model_name, self.revision)
        return self._enc.encode(encoder_input(list(X)))

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_enc"] = None
        return state

    def get_params(self, deep=True):
        return {"model_name": self.model_name, "revision": self.revision}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def deployable_embedding_pipeline(fitted: Pipeline, model_name: str, revision: str | None) -> Pipeline:
    return Pipeline([("encode", EncoderTransformer(model_name, revision))] + list(fitted.steps))
