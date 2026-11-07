import os
import subprocess
import sys

import joblib
import numpy as np
import pytest

from banking_nlp.evaluation import classification_metrics, paired_bootstrap_delta, bootstrap_indices, topk_accuracy
from banking_nlp.models import (LexicalRep, class_scores, deployable_lexical_pipeline, make_embedding_model,
                                make_lexical_model)
from banking_nlp.preprocessing import LEXICAL_VARIANTS, preprocess_lexical

LABELS = [f"label_{k}" for k in range(77)]


def test_vocabulary_is_fitted_on_fitting_texts_only(toy_data):
    fit, val = toy_data
    rep = LexicalRep("word", "L3_numbers", (1, 1))
    model = make_lexical_model(rep, "lr", 1.0)
    Xf = preprocess_lexical(fit.text.tolist(), LEXICAL_VARIANTS["L3_numbers"])
    model.fit(Xf, fit.label_id)
    vocab = model.named_steps["tfidf"].vocabulary_
    assert "heldoutonlyword" not in vocab
    model.predict(preprocess_lexical(["heldoutonlyword intent3"], LEXICAL_VARIANTS["L3_numbers"]))
    assert "heldoutonlyword" not in model.named_steps["tfidf"].vocabulary_


@pytest.mark.parametrize("clf", ["lr", "svm"])
@pytest.mark.parametrize("kind", ["word", "char", "word+char"])
def test_scores_shape_and_saved_pipeline_round_trip(toy_data, tmp_path, clf, kind):
    fit, val = toy_data
    rep = LexicalRep(kind, "L3_numbers", (1, 2), (2, 5))
    cfg = LEXICAL_VARIANTS["L3_numbers"]
    model = make_lexical_model(rep, clf, 1.0).fit(preprocess_lexical(fit.text.tolist(), cfg), fit.label_id)
    scores, stype = class_scores(model, preprocess_lexical(val.text.tolist(), cfg))
    assert scores.shape == (len(val), 77)
    assert stype.startswith("probability") == (clf == "lr")
    pipe = deployable_lexical_pipeline(model, rep, LABELS)
    joblib.dump(pipe, tmp_path / "m.joblib")
    reloaded = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(class_scores(reloaded, val.text.tolist())[0], scores)


@pytest.mark.parametrize("clf", ["lr", "svm"])
def test_embedding_classifier_contract(toy_data, fake_encoder, clf):
    fit, val = toy_data
    model = make_embedding_model(clf, 1.0).fit(fake_encoder.encode(fit.text), fit.label_id)
    scores, _ = class_scores(model, fake_encoder.encode(val.text))
    assert scores.shape == (len(val), 77) and scores.dtype.kind == "f"


def test_invalid_predictions_count_as_errors():
    y = np.array([0, 1, 2, 3])
    m = classification_metrics(y, [0, -1, None, 3])
    assert m["n"] == 4 and m["invalid_predictions"] == 2 and m["accuracy"] == 0.5


def test_topk_and_paired_delta():
    y = np.array([0, 1, 2])
    scores = np.array([[0.1, 0.9, 0.0], [0.2, 0.3, 0.5], [0.0, 0.1, 0.9]])
    assert topk_accuracy(y, scores, 1) == pytest.approx(1 / 3)
    assert topk_accuracy(y, scores, 2) == pytest.approx(1.0)
    idx = bootstrap_indices(3, 50, 0)
    d = paired_bootstrap_delta(y, np.array([0, 1, 2]), np.array([0, 1, 2]), idx)
    assert d["delta_macro_f1"] == 0 and d["n_disagreements"] == 0


def test_imports_have_no_side_effects(tmp_path):
    code = ("import sys, banking_nlp.data, banking_nlp.preprocessing, banking_nlp.models, "
            "banking_nlp.evaluation, banking_nlp.experiments, banking_nlp.artifacts; "
            "heavy = [m for m in ('spacy', 'sentence_transformers', 'torch') if m in sys.modules]; "
            "assert not heavy, heavy")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(__file__)))
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)
    assert list(tmp_path.iterdir()) == []
