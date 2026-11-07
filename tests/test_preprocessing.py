import importlib.util

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from banking_nlp.preprocessing import (LEXICAL_VARIANTS, LexicalPreprocessor, encoder_input,
                                       expand_contractions, light_normalize, preprocess_lexical,
                                       protected_words, regex_tokenize)

L2 = LEXICAL_VARIANTS["L2_contractions"]
L3 = LEXICAL_VARIANTS["L3_numbers"]


@pytest.mark.parametrize("text", ["Why can't I top up?", "why cant i top up", "Why cannot I top up?",
                                  "Why can’t I top up?", "WHY CAN'T I TOP UP"])
def test_negation_survives_all_spellings(text):
    assert preprocess_lexical([text], L2)[0] == "why can not i top up"


def test_sklearn_default_tokenizer_loses_negation():
    """Motivation for L1/L2: the default token_pattern turns can't into 'can'."""
    an = TfidfVectorizer().build_analyzer()
    assert an("Why can't I top up?") == ["why", "can", "top", "up"]
    assert "not" in preprocess_lexical(["Why can't I top up?"], L2)[0].split()


def test_hyphen_and_compound_variants_match():
    a, b, c = preprocess_lexical(["top-up failed", "top up failed", "topup failed"], L2)
    assert a == b == c == "top up failed"


def test_numbers_and_currency():
    out = preprocess_lexical(["I was charged £5 and €1,000.50 twice"], L3)[0]
    assert out == "i was charged £ <num> and € <num> twice"
    assert regex_tokenize("2 weeks") == ["2", "weeks"]  # single characters are kept


def test_light_normalize_keeps_case_and_punctuation():
    assert light_normalize("\nI can’t use  my card!\n\n") == "I can't use my card!"
    assert encoder_input(["What's  my PIN?"]) == ["What's my PIN?"]


def test_contractions():
    assert expand_contractions("I've been waiting, it's late and I'm sure") == \
        "I have been waiting, it late and I am sure"
    assert expand_contractions("I won't, didnt") == "I will not, did not"


def test_protected_stopwords():
    prot = protected_words(["top_up_failed", "why_verify_identity", "get_physical_card"])
    for w in ("not", "no", "why", "how", "top", "up", "get", "can"):
        assert w in prot
    out = preprocess_lexical(["Why can't I top up my card?"], LEXICAL_VARIANTS["L5_stop_protected"], prot)[0]
    assert out == "why can not top up card"  # "my" is not protected for these toy labels
    generic = preprocess_lexical(["Why can't I top up my card?"], LEXICAL_VARIANTS["L6_stop_sklearn"])[0]
    assert generic == "card"  # what the generic list destroys


@pytest.mark.skipif(importlib.util.find_spec("en_core_web_sm") is None, reason="spaCy model missing")
def test_lemmatization_keeps_negation():
    out = preprocess_lexical(["I was charged twice and didn't get my cards"], LEXICAL_VARIANTS["L4_lemma"])[0]
    assert "not" in out.split() and "charge" in out.split() and "card" in out.split()


def test_preprocessor_is_stateless():
    p = LexicalPreprocessor(config=L3.to_dict())
    before = p.transform(["My card arrived?"])
    p.fit(["completely different text 123"] * 5)
    assert p.transform(["My card arrived?"]) == before
