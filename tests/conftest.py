import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeEncoder:
    """Deterministic stand-in for MiniLM, used ONLY to test pipeline contracts
    (shapes, caching, row order). Never used for any reported number."""

    def __init__(self, dim: int = 32):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.hv = HashingVectorizer(n_features=dim, alternate_sign=True, norm="l2")
        self.info = {"model_name": "fake-hashing-encoder", "resolved_revision": "test",
                     "embedding_dim": dim, "max_seq_length": 512, "normalize_module": True}

    def encode(self, texts):
        X = self.hv.transform(list(texts)).toarray().astype(np.float32)
        X[np.linalg.norm(X, axis=1) == 0, 0] = 1.0
        return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.fixture
def fake_encoder():
    return FakeEncoder()


@pytest.fixture
def toy_data():
    """77-class toy corpus: each class has a distinctive token plus shared filler."""
    rng = np.random.default_rng(0)
    rows = []
    for k in range(77):
        for i in range(6):
            filler = " ".join(rng.choice(["please", "help", "my", "card", "account", "why"], 3))
            rows.append({"text": f"intent{k} token{k} {filler}", "label_id": k})
    df = pd.DataFrame(rows)
    fit = df.groupby("label_id").head(4)
    val = df.drop(fit.index)
    return fit.reset_index(drop=True), val.reset_index(drop=True)


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def has_raw_data():
    return (ROOT / "data" / "raw" / "train.csv").exists()
