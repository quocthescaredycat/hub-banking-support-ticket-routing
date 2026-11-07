"""Data contract for BANKING77: source provenance, label mapping, duplicate audit,
and duplicate-group-aware development partitions.

Source
------
The canonical CSV files published by PolyAI in
https://github.com/PolyAI-LDN/task-specific-datasets (``banking_data/``), pinned to
one commit and verified by SHA-256. The Hugging Face card
(https://huggingface.co/datasets/PolyAI/banking77) describes the same 10,003/3,080
split and the CC-BY-4.0 licence; we read the files from GitHub because that is the
original publication and because it can be pinned by commit.

Label IDs follow the order of ``categories.json`` in that commit. Label *names* are
the canonical key everywhere; never compare label IDs across different sources
without mapping by name first.

Test-set lock
-------------
``load_train`` never opens ``test.csv``. The test file is only checksummed during
development. ``load_test_texts`` (inputs, no labels) and ``load_test_labels``
(answers) exist for the final scoring stage and refuse to run unless the caller
passes ``i_confirm_configuration_is_frozen=True``.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import sha256_file, sha256_text

# --------------------------------------------------------------------------------------
# Source provenance
# --------------------------------------------------------------------------------------

SOURCE_REPO = "https://github.com/PolyAI-LDN/task-specific-datasets"
SOURCE_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_SUBDIR = "banking_data"
SOURCE_FILES = {
    # file name -> expected SHA-256 of the bytes at SOURCE_COMMIT
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
}
LICENSE = "CC-BY-4.0 (Casanueva et al., 2020; PolyAI)"
EXPECTED_COUNTS = {"train": 10003, "test": 3080, "labels": 77}


def raw_url(name: str) -> str:
    return (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
        f"{SOURCE_COMMIT}/{SOURCE_SUBDIR}/{name}"
    )


def download_source(raw_dir: str | Path, force: bool = False) -> dict:
    """Download the pinned files (if missing) and verify every checksum.

    Returns a source manifest. Raises ``ValueError`` on any checksum mismatch.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, expected in SOURCE_FILES.items():
        path = raw_dir / name
        if force or not path.exists():
            urllib.request.urlretrieve(raw_url(name), path)  # noqa: S310 (pinned https URL)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
        files[name] = {"url": raw_url(name), "sha256": actual, "bytes": path.stat().st_size}
    return {
        "dataset": "BANKING77",
        "repository": SOURCE_REPO,
        "commit": SOURCE_COMMIT,
        "license": LICENSE,
        "files": files,
        "note": "test.csv is checksummed only; it is not parsed during development.",
    }


def verify_source(raw_dir: str | Path) -> None:
    for name, expected in SOURCE_FILES.items():
        actual = sha256_file(Path(raw_dir) / name)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}")


# --------------------------------------------------------------------------------------
# Label mapping
# --------------------------------------------------------------------------------------

def load_label_names(raw_dir: str | Path) -> list[str]:
    """Canonical label names in the order of ``categories.json`` (= label IDs 0..76)."""
    with (Path(raw_dir) / "categories.json").open(encoding="utf-8") as fh:
        names = json.load(fh)
    if len(names) != EXPECTED_COUNTS["labels"] or len(set(names)) != len(names):
        raise ValueError("categories.json must contain 77 unique label names")
    return list(names)


def display_name(label_name: str) -> str:
    """Friendly display text; the canonical name is never modified."""
    return label_name.replace("_", " ").capitalize()


# --------------------------------------------------------------------------------------
# Duplicate keys
# --------------------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^\w\s]|_", flags=re.UNICODE)


def duplicate_key(text: str) -> str:
    """Exact-duplicate normalisation from the plan: NFKC, case folding, collapsed
    whitespace (and stripped ends). Punctuation and words are untouched."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def loose_duplicate_key(text: str) -> str:
    """Duplicate key with punctuation removed as well. Two requests that differ only
    in punctuation, case, or spacing (e.g. a trailing "?") get the same loose key."""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", duplicate_key(text))).strip()


def normalized_text_hash(text: str) -> str:
    return sha256_text(duplicate_key(text))[:16]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    # csv module, not pandas defaults: avoids "NA"/"null" texts becoming NaN and
    # handles the quoted multi-line requests in the source file.
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or set(rows[0]) != {"text", "category"}:
        raise ValueError(f"Unexpected columns in {path.name}")
    return rows


def _to_frame(rows: list[dict], split: str, label_names: list[str]) -> pd.DataFrame:
    label_to_id = {n: i for i, n in enumerate(label_names)}
    unknown = sorted({r["category"] for r in rows} - set(label_to_id))
    if unknown:
        raise ValueError(f"Unknown categories in {split}: {unknown[:5]}")
    df = pd.DataFrame(
        {
            "row_id": [f"{split}-{i:05d}" for i in range(len(rows))],
            "source_split": split,
            "raw_text": [r["text"] for r in rows],
            "label_name": [r["category"] for r in rows],
        }
    )
    df["label_id"] = df["label_name"].map(label_to_id).astype(int)
    df["normalized_text_hash"] = df["raw_text"].map(normalized_text_hash)
    return df


def load_train(raw_dir: str | Path) -> pd.DataFrame:
    """Official training rows with the plan's row contract. Never touches test.csv."""
    raw_dir = Path(raw_dir)
    labels = load_label_names(raw_dir)
    df = _to_frame(_read_csv(raw_dir / "train.csv"), "train", labels)
    if len(df) != EXPECTED_COUNTS["train"]:
        raise ValueError(f"Expected {EXPECTED_COUNTS['train']} training rows, got {len(df)}")
    return df


class TestSetLockedError(RuntimeError):
    pass


def _check_unlock(flag: bool) -> None:
    if not flag:
        raise TestSetLockedError(
            "The official test set is locked during development. Freeze the "
            "configuration first (artifacts/frozen/selection.json) and call the final "
            "scoring stage with i_confirm_configuration_is_frozen=True."
        )


def load_test_texts(raw_dir: str | Path, *, i_confirm_configuration_is_frozen: bool = False) -> pd.DataFrame:
    """Test inputs only (row_id, raw_text). Labels are dropped before returning."""
    _check_unlock(i_confirm_configuration_is_frozen)
    raw_dir = Path(raw_dir)
    df = _to_frame(_read_csv(raw_dir / "test.csv"), "test", load_label_names(raw_dir))
    return df[["row_id", "source_split", "raw_text", "normalized_text_hash"]].copy()


def load_test_labels(raw_dir: str | Path, *, i_confirm_configuration_is_frozen: bool = False) -> pd.DataFrame:
    """Test answers for the separate scoring function (row_id, label_name, label_id)."""
    _check_unlock(i_confirm_configuration_is_frozen)
    raw_dir = Path(raw_dir)
    df = _to_frame(_read_csv(raw_dir / "test.csv"), "test", load_label_names(raw_dir))
    if len(df) != EXPECTED_COUNTS["test"]:
        raise ValueError(f"Expected {EXPECTED_COUNTS['test']} test rows, got {len(df)}")
    return df[["row_id", "label_name", "label_id"]].copy()


# --------------------------------------------------------------------------------------
# Audit and duplicate groups
# --------------------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


NEAR_DUP_THRESHOLD = 0.90


def duplicate_groups(texts: list[str], near_dup_threshold: float | None = NEAR_DUP_THRESHOLD) -> np.ndarray:
    """Group IDs for duplicate-aware partitioning.

    Fixed rule, decided from the official-training audit before partitioning:
    rows are merged when they share an exact key, a loose key (case, spacing,
    punctuation or currency-symbol variants), or are near-verbatim copies
    (word 1-2-gram TF-IDF cosine >= 0.90 on duplicate keys). At >= 0.90 the
    audited pairs are template copies such as an added "please"; around 0.85
    genuine minimal pairs with different intents appear ("...fee for my card
    payment?" vs "...fee for my transfer?"), so the threshold is not lowered.
    Paraphrases are never grouped by intent.
    """
    uf = _UnionFind(len(texts))
    first_seen: dict[tuple[str, str], int] = {}
    for i, t in enumerate(texts):
        for kind, key in (("exact", duplicate_key(t)), ("loose", loose_duplicate_key(t))):
            j = first_seen.setdefault((kind, key), i)
            if j != i:
                uf.union(i, j)
    if near_dup_threshold is not None:
        for i, j, _ in near_duplicate_pairs(texts, threshold=near_dup_threshold):
            uf.union(i, j)
    roots = [uf.find(i) for i in range(len(texts))]
    # dense, order-stable group ids
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(r, len(remap)) for r in roots], dtype=int)


def near_duplicate_pairs(texts: list[str], threshold: float = 0.90, chunk: int = 1000) -> list[tuple[int, int, float]]:
    """Candidate near-duplicates for documentation only: cosine >= threshold between
    word 1-2-gram TF-IDF vectors of the duplicate keys (fitted on the given texts,
    i.e. official training only). Pairs already sharing a loose key are skipped."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    keys = [duplicate_key(t) for t in texts]
    loose = [loose_duplicate_key(t) for t in texts]
    X = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b").fit_transform(keys)
    pairs = []
    for start in range(0, X.shape[0], chunk):
        sims = (X[start:start + chunk] @ X.T).toarray()
        for local_i, row in enumerate(sims):
            i = start + local_i
            for j in np.nonzero(row >= threshold)[0]:
                if j > i and loose[i] != loose[j]:
                    pairs.append((i, int(j), float(row[j])))
    return pairs


def audit_train(df: pd.DataFrame, near_dup_threshold: float = NEAR_DUP_THRESHOLD) -> dict:
    """Descriptive audit of official training rows (counts, lengths, labels, duplicates)."""
    texts = df["raw_text"].tolist()
    n_words = df["raw_text"].str.split().str.len()
    n_chars = df["raw_text"].str.len()

    exact = defaultdict(list)
    loose = defaultdict(list)
    for i, t in enumerate(texts):
        exact[duplicate_key(t)].append(i)
        loose[loose_duplicate_key(t)].append(i)

    def summarise(groups: dict) -> dict:
        multi = {k: v for k, v in groups.items() if len(v) > 1}
        conflicts = {k: v for k, v in multi.items() if df["label_name"].iloc[v].nunique() > 1}
        return {
            "groups_with_duplicates": len(multi),
            "rows_in_duplicate_groups": int(sum(len(v) for v in multi.values())),
            "conflicting_label_groups": len(conflicts),
            "conflict_examples": [
                {"text": texts[v[0]], "labels": sorted(df["label_name"].iloc[v].unique().tolist()),
                 "row_ids": df["row_id"].iloc[v].tolist()}
                for v in list(conflicts.values())[:20]
            ],
            "duplicate_examples": [
                {"texts": [texts[i] for i in v][:3], "labels": df["label_name"].iloc[v].tolist()[:3]}
                for v in list(multi.values())[:10]
            ],
        }

    near = near_duplicate_pairs(texts, threshold=near_dup_threshold)
    near_same_label = sum(df["label_id"].iat[i] == df["label_id"].iat[j] for i, j, _ in near)
    label_counts = df["label_name"].value_counts()
    whitespace_issues = int(sum(t != t.strip() or "  " in t or "\n" in t for t in texts))
    non_ascii = defaultdict(int)
    for t in texts:
        for ch in t:
            if ord(ch) > 127:
                non_ascii[f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}"] += 1

    return {
        "rows": len(df),
        "labels_present": int(df["label_name"].nunique()),
        "empty_or_blank_texts": int(sum(not t.strip() for t in texts)),
        "texts_with_whitespace_irregularities": whitespace_issues,
        "multiline_texts": int(sum("\n" in t or "\r" in t for t in texts)),
        "non_ascii_characters": dict(sorted(non_ascii.items(), key=lambda kv: -kv[1])),
        "length_words": {k: float(v) for k, v in n_words.describe(percentiles=[.05, .5, .95]).items()},
        "length_chars": {k: float(v) for k, v in n_chars.describe(percentiles=[.05, .5, .95]).items()},
        "label_frequency": {
            "min": int(label_counts.min()), "max": int(label_counts.max()),
            "median": float(label_counts.median()),
            "least_frequent": label_counts.tail(5).to_dict(),
            "most_frequent": label_counts.head(5).to_dict(),
        },
        "exact_duplicates": summarise(exact),
        "loose_duplicates": summarise(loose),
        "near_duplicate_candidates": {
            "rule": f"word 1-2-gram TF-IDF cosine >= {near_dup_threshold} on duplicate keys, "
                    "pairs not already sharing a loose key",
            "pairs": len(near),
            "pairs_with_same_label": int(near_same_label),
            "decision": "merged into duplicate groups for partitioning (near-verbatim copies); "
                        "lower thresholds start pairing distinct intents, so they are not used",
            "examples": [
                {"a": texts[i], "b": texts[j], "cosine": round(s, 3),
                 "labels": [df["label_name"].iat[i], df["label_name"].iat[j]]}
                for i, j, s in sorted(near, key=lambda x: -x[2])[:15]
            ],
        },
    }


# --------------------------------------------------------------------------------------
# Development partitions
# --------------------------------------------------------------------------------------

PARTITIONS = ("fit", "model_val", "routing_val")


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    n_folds: int = 10          # 1 fold model_val, 1 fold routing_val, 8 folds fit
    model_val_fold: int = 0
    routing_val_fold: int = 1


def make_splits(df: pd.DataFrame, cfg: SplitConfig = SplitConfig()) -> pd.DataFrame:
    """~80/10/10 duplicate-group-aware stratified partition of official training."""
    from sklearn.model_selection import StratifiedGroupKFold

    out = df.copy()
    out["group_id"] = duplicate_groups(out["raw_text"].tolist())
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_of = np.full(len(out), -1, dtype=int)
    for fold, (_, idx) in enumerate(sgkf.split(out, out["label_id"], groups=out["group_id"])):
        fold_of[idx] = fold
    assert (fold_of >= 0).all()
    partition = np.where(fold_of == cfg.model_val_fold, "model_val",
                         np.where(fold_of == cfg.routing_val_fold, "routing_val", "fit"))
    out["partition"] = partition
    check_splits(out)
    return out


def check_splits(df: pd.DataFrame, n_labels: int = 77) -> None:
    """Contract: groups never straddle partitions; every partition has all labels;
    no test rows are present."""
    if (df["source_split"] != "train").any():
        raise AssertionError("Development partitions must contain official training rows only")
    straddle = df.groupby("group_id")["partition"].nunique()
    if (straddle > 1).any():
        raise AssertionError(f"{int((straddle > 1).sum())} duplicate groups straddle partitions")
    for p in PARTITIONS:
        n = df.loc[df["partition"] == p, "label_id"].nunique()
        if n != n_labels:
            raise AssertionError(f"Partition {p} has {n} labels, expected {n_labels}")


def split_summary(df: pd.DataFrame) -> dict:
    counts = df["partition"].value_counts().reindex(PARTITIONS).astype(int)
    per_label = df.groupby(["partition", "label_name"]).size().unstack(0).reindex(columns=PARTITIONS)
    near = near_duplicate_pairs(df["raw_text"].tolist(), threshold=0.85)
    part = df["partition"].to_numpy()
    cross = [(i, j) for i, j, _ in near if part[i] != part[j]]
    sizes = df.groupby("group_id").size()
    return {
        "counts": counts.to_dict(),
        "fractions": {k: round(v / len(df), 4) for k, v in counts.items()},
        "requested_fractions": {"fit": 0.8, "model_val": 0.1, "routing_val": 0.1},
        "per_label_min": per_label.min().astype(int).to_dict(),
        "per_label_max": per_label.max().astype(int).to_dict(),
        "duplicate_groups": int(df["group_id"].nunique()),
        "multi_row_groups": int((sizes > 1).sum()),
        "largest_group_size": int(sizes.max()),
        "pairs_cosine_0.85_or_more_across_partitions": len(cross),
        "note": "pairs between 0.85 and 0.90 are allowed to cross partitions; many are "
                "minimal pairs with different intents",
    }
