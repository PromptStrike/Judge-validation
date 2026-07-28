"""Agreement analysis — the intellectual core of judge validation.

Pure functions only. No I/O, no network. Everything here is testable
with hand-built fixtures, which is why it's built first.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence, Hashable


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



class AgreementResult:
    n: int
    raw_agreement: float          # % of items where both agree
    cohens_kappa: float           # chance-corrected agreement
    confusion: dict               # {(a_verdict, b_verdict): count}
    disagreements: list[str]      # sample_ids where they differ


def raw_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Fraction of items where two raters gave the same verdict."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("empty input")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Chance-corrected agreement.

    kappa = (po - pe) / (1 - pe)
      po = observed agreement
      pe = agreement expected by chance, given each rater's base rates

    Why this and not raw agreement: if 95% of samples are "not success",
    a rater that always says "not success" scores 0.95 raw agreement while
    being useless. Kappa corrects for that and returns ~0.
    """
    po = raw_agreement(a, b)

    n = len(a)
    count_a, count_b = Counter(a), Counter(b)
    labels = set(count_a) | set(count_b)

    # probability both pick the same label purely by chance
    pe = sum((count_a[l] / n) * (count_b[l] / n) for l in labels)

    if pe == 1.0:                     # both raters used exactly one label
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def confusion_matrix(a: Sequence[Hashable], b: Sequence[Hashable]) -> dict:
    """Counts of every (rater_a_verdict, rater_b_verdict) pair.

    This is where PATTERNS live — e.g. judge=success/human=not-success
    dominating tells you the judge over-reports, which is actionable.
    A single agreement number never tells you that.
    """
    return dict(Counter(zip(a, b)))


def analyse(sample_ids: Sequence[str],
            rater_a: Sequence[Hashable],
            rater_b: Sequence[Hashable]) -> AgreementResult:
    """Full comparison of two raters over the same samples."""
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


def interpret(judge_vs_human: AgreementResult,
              human_vs_human: AgreementResult | None = None) -> str:
    """Turn numbers into a verdict a human can act on.

    The ceiling logic: a judge can't beat the consistency of the humans
    defining the task. If it's close to the ceiling, the construct is the
    bottleneck, not the judge.
    """
    k = judge_vs_human.cohens_kappa
    if human_vs_human is None:
        return f"kappa={k:.2f} (no human-human ceiling measured — trust unknown)"

    ceiling = human_vs_human.cohens_kappa
    if ceiling < 0.6:
        return (f"kappa={k:.2f}, ceiling={ceiling:.2f}. "
                "Humans disagree substantially — fix the DEFINITION, not the judge.")
    if k >= ceiling - 0.05:
        return (f"kappa={k:.2f}, ceiling={ceiling:.2f}. "
                "Judge is at the ceiling; residual error is construct ambiguity.")
    return (f"kappa={k:.2f}, ceiling={ceiling:.2f}. "
            "Judge underperforms the ceiling — the judge is improvable.")
