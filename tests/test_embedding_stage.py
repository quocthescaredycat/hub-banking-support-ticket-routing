"""Integration test of the C3/C4 stage with a fake encoder in a temporary copy of the
project. It checks the code path (caching, row order, artifacts), not MiniLM quality."""

import json
import shutil

import joblib
import numpy as np
import pytest

from banking_nlp import experiments as E


@pytest.fixture
def temp_project(root, tmp_path):
    if not (root / "artifacts" / "data" / "splits.csv").exists():
        pytest.skip("partitions not frozen yet")
    for sub in ("configs", "data/raw", "artifacts/data"):
        shutil.copytree(root / sub, tmp_path / sub)
    cfg = json.loads((tmp_path / "configs/experiment.json").read_text())
    cfg["embedding"]["C_grid"] = [1, 10]  # keep the test fast
    (tmp_path / "configs/experiment.json").write_text(json.dumps(cfg))
    return tmp_path


def test_embedding_stage_with_fake_encoder(temp_project, fake_encoder):
    grid = E.run_embeddings(temp_project, encoder=fake_encoder, verbose=False)
    assert set(grid.family) == {"C3", "C4"}
    for fam in ("C3", "C4"):
        cand = json.loads((temp_project / f"artifacts/dev_results/candidate_{fam}.json").read_text())
        assert cand["n_fit_rows"] == 8003 and cand["val_metrics"]["n"] == 1000
        scores = np.load(temp_project / f"artifacts/predictions/model_val/{fam}_scores.npy")
        assert scores.shape == (1000, 77)
        pipe = joblib.load(temp_project / cand["artifact"])
        assert pipe.steps[0][0] == "encode"  # saved pipeline starts from raw text


def test_embedding_cache_rejects_wrong_row_order(temp_project, fake_encoder):
    E.run_embeddings(temp_project, encoder=fake_encoder, verbose=False)
    meta_path = temp_project / "artifacts/embeddings/minilm_dev_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["row_order_checksum"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="row order"):
        E.run_embeddings(temp_project, encoder=fake_encoder, verbose=False)


def test_final_stage_refuses_without_freeze(temp_project):
    from banking_nlp.data import TestSetLockedError

    with pytest.raises(TestSetLockedError):
        E.predict_test(temp_project)
