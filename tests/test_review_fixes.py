import numpy as np

from banking_nlp.evaluation import _resample_macro_f1, macro_f1
from banking_nlp.routing import choose_threshold, difficulty_slice


def test_resample_macro_f1_ignores_labels_absent_from_answers_and_predictions():
    y = np.array([0, 0, 1, 1])
    perfect = y.copy()
    assert _resample_macro_f1(y, perfect) == 1.0
    assert macro_f1(y, perfect) < 0.03          # fixed 77-label list: 75 phantom zeros
    # a label predicted but absent from the answers still counts as 0
    assert _resample_macro_f1(y, np.array([0, 0, 1, 5])) < 1.0


def test_threshold_rule_coverage_then_larger_threshold():
    margins = np.linspace(0, 1, 300)
    correct = margins > 0.1                      # errors only at low margins
    res = choose_threshold(margins, correct, max_error=0.05, min_routed=100)
    routed = margins >= res["threshold"]
    assert 1 - correct[routed].mean() <= 0.05 and routed.sum() >= 100
    # nothing qualifies -> review everything
    assert choose_threshold(margins, np.zeros(300, bool))["threshold"] is None


def test_difficulty_slice_picks_k_per_intent():
    fit = ["card arrived", "card delivery", "top up failed", "top up pending"]
    fy = [0, 0, 1, 1]
    test = ["card arrived", "zzz unknown", "card qqq", "top up failed", "www", "top up"]
    ty = [0, 0, 0, 1, 1, 1]
    keep = difficulty_slice(fit, fy, test, ty, per_intent=1)
    assert list(keep) == [1, 4]
