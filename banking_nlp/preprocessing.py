"""Text preprocessing for the two representation families.

Lexical branch (TF-IDF): the steps below are applied in this order. Each optional
step is a switch in :class:`LexicalConfig`, so every variant in the ablation is a
named, reproducible configuration.

  1. light normalisation   Unicode NFKC; typographic quotes/dashes/ellipsis mapped
                           to ASCII; zero-width characters removed; whitespace
                           (including the newlines and no-break spaces found in the
                           source) collapsed.
  2. contraction handling  negation-preserving expansion: "can't"/"cant"/"cannot" ->
                           "can not", "didn't"/"didnt" -> "did not", "won't" -> "will
                           not", plus "'re/'ve/'ll/'m/'d"; possessive/"is" clitic "'s"
                           is dropped. Also "topup(s)" -> "top up(s)".
  3. lemmatisation         (optional) spaCy ``en_core_web_sm`` POS-aware lemmas,
                           computed on case-preserved text for better tagging.
  4. case folding
  5. tokenisation          regex tokenizer that keeps single-character tokens,
                           numbers and currency symbols, splits hyphenated
                           compounds ("top-up" -> "top up"), and drops punctuation.
  6. number placeholder    (optional) numeric tokens -> "<num>"; currency symbols
                           are kept as their own tokens.
  7. stop-word removal     (optional, for the ablation only) either scikit-learn's
                           generic list, or that list minus a protected set
                           (negations, wh-words, modals, words in label names).

Every step is a stateless, row-wise function of the text: nothing is fitted, so the
preprocessed strings can be cached for all rows without leakage. Only the TF-IDF
vocabulary/IDF (in ``models.py``) is fitted, and only on the fitting partition.

Encoder branch (MiniLM): only step 1. The sentence encoder has its own WordPiece
tokenizer and was pretrained on natural text, so lemmas, removed stop words or
placeholders would move inputs away from its training distribution.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Literal

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --------------------------------------------------------------------------------------
# 1. Light normalisation (shared by both branches)
# --------------------------------------------------------------------------------------

_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ",
    "​": None, "‌": None, "‍": None, "﻿": None,
})
_WS = re.compile(r"\s+")


def light_normalize(text: str) -> str:
    """NFKC + ASCII quotes/dashes + whitespace collapse. Case, words, numbers and
    punctuation are preserved. This is the only step applied to encoder inputs."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP)
    return _WS.sub(" ", text).strip()


# --------------------------------------------------------------------------------------
# 2. Contractions and negation
# --------------------------------------------------------------------------------------

# Irregular negations first, then the generic n't rule.
_SPECIAL_NEG = [
    (r"\bcan'?t\b", "can not"),
    (r"\bcannot\b", "can not"),
    (r"\bwon'?t\b", "will not"),
    (r"\bshan'?t\b", "shall not"),
    (r"\bain'?t\b", "is not"),
]
# Apostrophe-less negations that occur in user-typed requests ("dont", "didnt").
_BARE_NEG = ["dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "havent",
             "hasnt", "hadnt", "couldnt", "wouldnt", "shouldnt", "mustnt", "neednt"]
_CLITICS = [
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'ve\b", " have"),
    (r"'ll\b", " will"),
    (r"'m\b", " am"),
    (r"'d\b", " would"),
    (r"'s\b", ""),  # possessive or "is/has": ambiguous, carries little intent signal
]
_DOMAIN_VARIANTS = [
    (r"\btopups\b", "top ups"),
    (r"\btopup\b", "top up"),
]
_SPECIAL_NEG_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _SPECIAL_NEG]
_BARE_NEG_RE = re.compile(r"\b(" + "|".join(_BARE_NEG) + r")\b", re.IGNORECASE)
_CLITIC_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _CLITICS]
_DOMAIN_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _DOMAIN_VARIANTS]


def expand_contractions(text: str) -> str:
    """Negation-preserving contraction expansion (expects light-normalised text)."""
    for pat, rep in _SPECIAL_NEG_RE:
        text = pat.sub(rep, text)
    text = _BARE_NEG_RE.sub(lambda m: m.group(1)[:-2] + " not", text)
    for pat, rep in _CLITIC_RE:
        text = pat.sub(rep, text)
    for pat, rep in _DOMAIN_RE:
        text = pat.sub(rep, text)
    return _WS.sub(" ", text).strip()


# --------------------------------------------------------------------------------------
# 3. Lemmatisation (spaCy, lazily loaded, cached per input string)
# --------------------------------------------------------------------------------------

SPACY_MODEL = "en_core_web_sm"
_NLP = None
_LEMMA_CACHE: dict[str, str] = {}


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy  # imported lazily so the package imports without spaCy

        _NLP = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    return _NLP


def lemmatize_texts(texts: list[str], batch_size: int = 256) -> list[str]:
    """POS-aware lemmas joined by spaces. Results are cached by input string."""
    todo = [t for t in dict.fromkeys(texts) if t not in _LEMMA_CACHE]
    if todo:
        nlp = _get_nlp()
        for src, doc in zip(todo, nlp.pipe(todo, batch_size=batch_size)):
            _LEMMA_CACHE[src] = " ".join(tok.lemma_ for tok in doc if not tok.is_space)
    return [_LEMMA_CACHE[t] for t in texts]


# --------------------------------------------------------------------------------------
# 5. Tokenisation, 6. numbers, 7. stop words
# --------------------------------------------------------------------------------------

NUM_TOKEN = "<num>"
# currency symbol | number (with , or . groups) | word (letters, optional internal
# apostrophe so that "can't" survives as one token when expansion is switched off)
TOKEN_RE = re.compile(r"[£$€¥]|\d+(?:[.,]\d+)*|[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
_NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)*$")


def regex_tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "nowhere",
             "neither", "cannot", "without"}
WH_WORDS = {"what", "why", "how", "when", "where", "which", "who", "whom", "whose", "whether"}
MODALS = {"can", "could", "will", "would", "should", "may", "might", "must", "shall"}


def protected_words(label_names: Iterable[str]) -> set[str]:
    """Stop words that must not be removed: negations, wh-words, modals, and every
    stop word that appears inside a canonical label name (e.g. "top", "up", "get",
    "amount", "not", "why", "into", "via")."""
    label_tokens = {w for name in label_names for w in name.lower().split("_")}
    return NEGATIONS | WH_WORDS | MODALS | (set(ENGLISH_STOP_WORDS) & label_tokens)


@dataclass(frozen=True)
class LexicalConfig:
    """One named lexical preprocessing variant."""

    name: str
    tokenizer: Literal["sklearn_default", "regex"] = "regex"
    contractions: bool = True
    lemmatize: bool = False
    numbers: Literal["keep", "placeholder"] = "keep"
    stopwords: Literal["none", "protected", "sklearn"] = "none"
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def preprocess_lexical(texts: list[str], cfg: LexicalConfig,
                       protected: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Raw texts -> space-joined token strings for the TF-IDF vectorizer.

    For ``tokenizer="sklearn_default"`` the raw text is returned unchanged so that the
    vectorizer's own lowercasing and default ``token_pattern`` apply (reference L0).
    """
    if cfg.tokenizer == "sklearn_default":
        return list(texts)
    out = [light_normalize(t) for t in texts]
    if cfg.contractions:
        out = [expand_contractions(t) for t in out]
    if cfg.lemmatize:
        out = lemmatize_texts(out)
    out = [t.casefold() for t in out]
    if cfg.stopwords == "sklearn":
        stop = set(ENGLISH_STOP_WORDS)
    elif cfg.stopwords == "protected":
        stop = set(ENGLISH_STOP_WORDS) - set(protected)
    else:
        stop = set()
    result = []
    for t in out:
        toks = regex_tokenize(t)
        if cfg.numbers == "placeholder":
            toks = [NUM_TOKEN if _NUMERIC_RE.match(tok) else tok for tok in toks]
        if stop:
            toks = [tok for tok in toks if tok not in stop]
        result.append(" ".join(toks))
    return result


def encoder_input(texts: list[str]) -> list[str]:
    """Inputs for the pretrained sentence encoder: light normalisation only."""
    return [light_normalize(t) for t in texts]


class LexicalPreprocessor(BaseEstimator, TransformerMixin):
    """Stateless sklearn transformer wrapping :func:`preprocess_lexical`, so that a
    saved pipeline accepts raw request text. ``fit`` learns nothing."""

    def __init__(self, config: dict | None = None, protected: tuple[str, ...] = ()):
        self.config = config
        self.protected = protected

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X):
        cfg = LexicalConfig(**(self.config or {"name": "default"}))
        return preprocess_lexical(list(X), cfg, protected=set(self.protected))


# --------------------------------------------------------------------------------------
# Named variants used by the ablation (cumulative where it makes sense)
# --------------------------------------------------------------------------------------

LEXICAL_VARIANTS: dict[str, LexicalConfig] = {
    "L0_sklearn_default": LexicalConfig(
        "L0_sklearn_default", tokenizer="sklearn_default", contractions=False,
        description="Reference: TfidfVectorizer defaults on raw text (lowercase, "
                    "token_pattern drops 1-char tokens, splits can't -> can + t)."),
    "L1_normalize_tokenize": LexicalConfig(
        "L1_normalize_tokenize", contractions=False,
        description="Light normalisation + case folding + regex tokenizer."),
    "L2_contractions": LexicalConfig(
        "L2_contractions",
        description="L1 + negation-preserving contraction expansion."),
    "L3_numbers": LexicalConfig(
        "L3_numbers", numbers="placeholder",
        description="L2 + number placeholder."),
    "L4_lemma": LexicalConfig(
        "L4_lemma", numbers="placeholder", lemmatize=True,
        description="L3 + spaCy POS-aware lemmatisation."),
    "L5_stop_protected": LexicalConfig(
        "L5_stop_protected", numbers="placeholder", stopwords="protected",
        description="L3 + stop-word removal with protected negations/wh/modals/label words."),
    "L6_stop_sklearn": LexicalConfig(
        "L6_stop_sklearn", numbers="placeholder", stopwords="sklearn",
        description="L3 + scikit-learn's generic stop-word list (removes not/top/up/why...)."),
}
# Character n-grams are a tokenisation-granularity choice applied to the L3 string.
CHAR_BASE_VARIANT = "L3_numbers"
