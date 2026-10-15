import math
import subprocess
import sys

import pytest

from trial_matcher.embed import (
    BASIC,
    WITH_INCLUSION,
    SentenceTransformerEncoder,
    build_trial_text,
    text_hash,
    vector_literal,
)


def test_basic_text_has_title_conditions_interventions():
    text = build_trial_text(BASIC, "A Study of Drug X.", ["Breast Cancer", "HER2"], ["Drug X", "Placebo"])
    assert text == "A Study of Drug X. Conditions: Breast Cancer; HER2. Interventions: Drug X; Placebo"


def test_missing_parts_are_left_out():
    assert build_trial_text(BASIC, "Only a title", [], []) == "Only a title"
    assert build_trial_text(BASIC, "T", ["C"], []) == "T. Conditions: C"


def test_basic_ignores_inclusion_criteria():
    with_crit = build_trial_text(BASIC, "T", ["C"], ["I"], ["Age 18+"])
    assert with_crit == build_trial_text(BASIC, "T", ["C"], ["I"])


def test_with_inclusion_adds_first_five_criteria_flattened():
    criteria = ["HER2-negative", "Stage IV\n- measurable disease\n- ECOG 0-1", "c3", "c4", "c5", "c6-not-used"]
    text = build_trial_text(WITH_INCLUSION, "T", ["C"], [], criteria)
    assert text.endswith("Inclusion: HER2-negative Stage IV - measurable disease - ECOG 0-1 c3 c4 c5")
    assert "c6-not-used" not in text


def test_with_inclusion_but_no_criteria_equals_basic():
    assert build_trial_text(WITH_INCLUSION, "T", ["C"], ["I"]) == build_trial_text(BASIC, "T", ["C"], ["I"])


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        build_trial_text("fancy", "T", [], [])


def test_text_hash_is_stable_and_sensitive():
    assert text_hash("abc") == text_hash("abc")
    assert text_hash("abc") != text_hash("abd")
    assert len(text_hash("abc")) == 64


def test_vector_literal_format():
    assert vector_literal([0.5, -1, 0.123456789123]) == "[0.5,-1,0.12345679]"


def test_importing_embed_does_not_load_torch():
    code = "import sys, trial_matcher.embed; print('torch' in sys.modules or 'sentence_transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": "src"})
    assert out.stdout.strip() == "False", out.stderr


@pytest.mark.slow
def test_real_model_gives_384_unit_vectors_and_similar_texts_are_closer():
    # Loads the real bge-small model (~130 MB download the first time): pytest -m slow
    enc = SentenceTransformerEncoder("BAAI/bge-small-en-v1.5")
    a, b, c = enc.encode(["metastatic breast cancer", "advanced breast carcinoma", "knee replacement surgery"])
    assert len(a) == 384
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-4)
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert dot(a, b) > dot(a, c)
