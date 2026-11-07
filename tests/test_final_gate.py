"""Gate before the test set, pooling-config parsing, and the test-uncertainty summary."""

import json
import shutil

import numpy as np
import pytest

from banking_nlp import experiments as E
from banking_nlp.artifacts import sha256_file
from banking_nlp import data as D
from banking_nlp.models import pooling_modes


@pytest.fixture
def frozen_project(root, tmp_path):
    if not (root / "data" / "raw" / "test.csv").exists():
        pytest.skip("raw data not downloaded")
    for sub in ("configs", "data/raw", "artifacts/data"):
        shutil.copytree(root / sub, tmp_path / sub)
    (tmp_path / "artifacts/frozen").mkdir(parents=True)
    (tmp_path / "artifacts/frozen/selection.json").write_text(json.dumps({"candidates": {}}))
    return tmp_path


def test_final_refuses_without_routing(frozen_project):
    with pytest.raises(D.TestSetLockedError, match="routing.json"):
        E.predict_test(frozen_project, i_confirm_configuration_is_frozen=True)


def test_final_refuses_routing_from_another_selection(frozen_project):
    (frozen_project / "artifacts/frozen/routing.json").write_text(json.dumps({"selection_sha256": "0" * 64}))
    with pytest.raises(D.TestSetLockedError, match="current selection.json"):
        E.predict_test(frozen_project, i_confirm_configuration_is_frozen=True)


def test_final_still_needs_explicit_confirmation(frozen_project):
    sel = frozen_project / "artifacts/frozen/selection.json"
    (frozen_project / "artifacts/frozen/routing.json").write_text(json.dumps({"selection_sha256": sha256_file(sel)}))
    with pytest.raises(D.TestSetLockedError):
        E.predict_test(frozen_project)  # flag not given
    assert not (frozen_project / "artifacts/predictions/test").exists()


@pytest.mark.parametrize("cfg, expected", [
    ({"embedding_dimension": 384, "pooling_mode": "mean", "include_prompt": True}, ["mean"]),
    ({"pooling_mode": ["cls", "mean"]}, ["cls", "mean"]),
    ({"pooling_mode": ("max",)}, ["max"]),
    ({"word_embedding_dimension": 384, "pooling_mode_cls_token": False,
      "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False}, ["mean_tokens"]),
])
def test_pooling_modes_both_config_formats(cfg, expected):
    assert pooling_modes(cfg) == expected


def test_bootstrap_test_uncertainty_pairs_and_direction():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 77, 400)
    good = y.copy()
    bad = np.where(rng.random(400) < 0.3, (y + 1) % 77, y)
    out = E.bootstrap_test_uncertainty(y, {"C1": bad, "C2": good, "C3": good}, {"n_boot": 50, "seed": 42})
    assert set(out["test_ci"]) == {"C1", "C2", "C3"}
    assert set(out["test_paired"]) == {"C2-C1", "C3-C1", "C3-C2"}
    assert out["test_paired"]["C2-C1"]["delta_macro_f1"] > 0       # "Cj-Ci" means Cj minus Ci
    assert out["test_paired"]["C3-C2"]["n_disagreements"] == 0
