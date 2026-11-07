import json

import pandas as pd
import pytest

from banking_nlp import data as D
from banking_nlp.artifacts import sha256_file


def test_duplicate_keys():
    assert D.duplicate_key("\n  Where can I   WITHDRAW money?\n") == "where can i withdraw money?"
    assert D.loose_duplicate_key("Where can I withdraw money?") == D.loose_duplicate_key("where can i withdraw money")
    # loose key must not merge different words
    assert D.loose_duplicate_key("card payment fee") != D.loose_duplicate_key("transfer fee")


def test_test_set_is_locked(root, has_raw_data):
    if not has_raw_data:
        pytest.skip("raw data not downloaded")
    with pytest.raises(D.TestSetLockedError):
        D.load_test_texts(root / "data" / "raw")
    with pytest.raises(D.TestSetLockedError):
        D.load_test_labels(root / "data" / "raw")


def test_label_mapping_round_trip(root, has_raw_data):
    if not has_raw_data:
        pytest.skip("raw data not downloaded")
    names = D.load_label_names(root / "data" / "raw")
    assert len(names) == 77 and len(set(names)) == 77
    mapping = json.loads((root / "artifacts" / "data" / "label_mapping.json").read_text())
    assert mapping["label_names"] == names
    assert all(names[mapping["label_to_id"][n]] == n for n in names)
    assert "Refund_not_showing_up" in names  # original capitalisation preserved


def test_source_checksums(root, has_raw_data):
    if not has_raw_data:
        pytest.skip("raw data not downloaded")
    D.verify_source(root / "data" / "raw")


def test_frozen_splits_contract(root):
    path = root / "artifacts" / "data" / "splits.csv"
    if not path.exists():
        pytest.skip("partitions not frozen yet")
    df = pd.read_csv(path, keep_default_na=False)
    summary = json.loads((root / "artifacts" / "data" / "split_summary.json").read_text())
    assert sha256_file(path) == summary["splits_csv_sha256"]
    assert len(df) == 10003 and set(df.source_split) == {"train"}
    D.check_splits(df)  # groups disjoint, 77 labels per partition
    frac = df.partition.value_counts(normalize=True)
    assert abs(frac["fit"] - 0.8) < 0.01 and abs(frac["model_val"] - 0.1) < 0.01
    # every row with the same exact duplicate key sits in one partition
    assert (df.groupby("normalized_text_hash").partition.nunique() == 1).all()
