"""Tests for agreement analysis.

No API, no I/O — pure functions with hand-built fixtures.
"""
import pytest
from jval.analysis import raw_agreement, cohens_kappa, confusion_matrix, analyse


def test_perfect_agreement():
    a = ["success", "not_success", "success"]
    assert raw_agreement(a, a) == 1.0
    assert cohens_kappa(a, a) == 1.0


def test_useless_judge_exposed_by_kappa():
    """A judge that always says 'success' against a skewed human set.

    This is THE case kappa exists for: raw agreement can look plausible
    while the judge carries no information.
    """
    judge = ["success"] * 20
    human = ["success"] * 2 + ["not_success"] * 18

    assert raw_agreement(judge, human) == pytest.approx(0.10)
    assert cohens_kappa(judge, human) == pytest.approx(0.0, abs=0.01)


def test_skewed_classes_inflate_raw_agreement():
    """Judge always says 'not_success'; humans almost always agree.

    Raw agreement = 0.95 (looks great). Kappa ~ 0 (it's useless).
    """
    judge = ["not_success"] * 20
    human = ["not_success"] * 19 + ["success"]

    assert raw_agreement(judge, human) == pytest.approx(0.95)
    assert cohens_kappa(judge, human) == pytest.approx(0.0, abs=0.01)


def test_confusion_shows_direction_of_error():
    judge = ["success", "success", "not_success"]
    human = ["not_success", "not_success", "not_success"]

    conf = confusion_matrix(judge, human)
    assert conf[("success", "not_success")] == 2   # judge over-reports
    assert conf[("not_success", "not_success")] == 1


def test_analyse_collects_disagreements():
    ids = ["s1", "s2", "s3"]
    judge = ["success", "success", "not_success"]
    human = ["success", "not_success", "not_success"]

    r = analyse(ids, judge, human)
    assert r.n == 3
    assert r.disagreements == ["s2"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        raw_agreement(["a", "b"], ["a"])


def test_empty_raises():
    with pytest.raises(ValueError):
        raw_agreement([], [])
