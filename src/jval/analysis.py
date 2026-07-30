"""Agreement analysis — the intellectual core of judge validation.
Pure functions only. No I/O, no network. Testable with hand-built fixtures.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence, Hashable


@dataclass(frozen=True)
class AgreementResult:
    n: int
    raw_agreement: float
    cohens_kappa: float
    confusion: dict
    disagreements: list


def raw_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("empty input")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    po = raw_agreement(a, b)
    n = len(a)
    count_a, count_b = Counter(a), Counter(b)
    labels = set(count_a) | set(count_b)
    pe = sum((count_a[l] / n) * (count_b[l] / n) for l in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def confusion_matrix(a: Sequence[Hashable], b: Sequence[Hashable]) -> dict:
    return dict(Counter(zip(a, b)))


def analyse(sample_ids: Sequence[str],
            rater_a: Sequence[Hashable],
            rater_b: Sequence[Hashable]) -> AgreementResult:
    if not (len(sample_ids) == len(rater_a) == len(rater_b)):
        raise ValueError("sample_ids and both rater sequences must align")
    return AgreementResult(
        n=len(sample_ids),
        raw_agreement=raw_agreement(rater_a, rater_b),
        cohens_kappa=cohens_kappa(rater_a, rater_b),
        confusion=confusion_matrix(rater_a, rater_b),
        disagreements=[sid for sid, x, y in zip(sample_ids, rater_a, rater_b)
                       if x != y],
    )


@dataclass(frozen=True)
class ValidationResult:
    decidable: AgreementResult
    n_ambiguous: int
    judge_on_ambiguous: dict


def validate(sample_ids, judge_verdicts, human_verdicts,
             ambiguous_label: str = "ambiguous") -> ValidationResult:
    """Split human-ambiguous cases out of the agreement calculation.
    Kappa is computed only where the human was decisive."""
    if not (len(sample_ids) == len(judge_verdicts) == len(human_verdicts)):
        raise ValueError("inputs must align")

    dec_ids, dec_judge, dec_human = [], [], []
    amb_judge_counts: dict = {}

    for sid, jv, hv in zip(sample_ids, judge_verdicts, human_verdicts):
        if hv == ambiguous_label:
            amb_judge_counts[jv] = amb_judge_counts.get(jv, 0) + 1
        else:
            dec_ids.append(sid)
            dec_judge.append(jv)
            dec_human.append(hv)

    if not dec_ids:
        raise ValueError("no decidable samples (all human labels ambiguous)")

    return ValidationResult(
        decidable=analyse(dec_ids, dec_judge, dec_human),
        n_ambiguous=sum(amb_judge_counts.values()),
        judge_on_ambiguous=amb_judge_counts,
    )
