"""Reliability report — ties labels + verdicts + analysis into a verdict
on whether a judge can be trusted. No API — pure analysis over stored data."""
from __future__ import annotations

from jval.store import Store
from jval.analysis import analyse, AgreementResult, validate, ValidationResult
from jval.judges.llm_judge import PARSE_ERROR


def judge_vs_human(store: Store, judge_id: str, judge_version: str,
                   labeler_id: str) -> AgreementResult:
    judge_map = {k: v for k, v in
                 store.judge_verdicts(judge_id, judge_version).items()
                 if v != PARSE_ERROR}
    human_map = store.latest_labels(labeler_id)
    ls_judge, ls_human = store.build_label_sets(
        (judge_id, judge_map), (labeler_id, human_map),
        kind_a="judge", kind_b="human")
    return analyse(ls_judge.sample_ids, ls_judge.verdicts, ls_human.verdicts)


def human_vs_human(store: Store, labeler_a: str, labeler_b: str) -> AgreementResult:
    a = store.latest_labels(labeler_a)
    b = store.latest_labels(labeler_b)
    ls_a, ls_b = store.build_label_sets(
        (labeler_a, a), (labeler_b, b), kind_a="human", kind_b="human")
    return analyse(ls_a.sample_ids, ls_a.verdicts, ls_b.verdicts)


def format_report(judge_result: AgreementResult, judge_label: str = "judge",
                  ceiling: AgreementResult | None = None) -> str:
    lines = ["=" * 60,
             f"JUDGE RELIABILITY REPORT — {judge_label}",
             "=" * 60,
             f"  samples compared : {judge_result.n}",
             f"  raw agreement    : {judge_result.raw_agreement:.2f}",
             f"  Cohen's kappa    : {judge_result.cohens_kappa:.2f}"]
    if ceiling is not None:
        lines.append(f"  human-human ceil : {ceiling.cohens_kappa:.2f} (n={ceiling.n})")
    k = judge_result.cohens_kappa
    lines.append("-" * 60)
    if ceiling is not None and ceiling.cohens_kappa < 0.6:
        lines.append("  VERDICT: Humans disagree substantially. Fix the")
        lines.append("           DEFINITION, not the judge.")
    elif ceiling is not None and k >= ceiling.cohens_kappa - 0.05:
        lines.append("  VERDICT: Judge is at the human ceiling — trustworthy.")
    elif k >= 0.8:
        lines.append("  VERDICT: Strong agreement — judge is trustworthy.")
    elif k >= 0.6:
        lines.append("  VERDICT: Moderate — usable but verify edge cases.")
    else:
        lines.append("  VERDICT: Weak agreement — do NOT trust this judge.")
    if judge_result.confusion:
        lines.append("-" * 60)
        lines.append("  Disagreement patterns (judge, human): count")
        for (jv, hv), c in sorted(judge_result.confusion.items(), key=lambda x: -x[1]):
            if jv != hv:
                lines.append(f"    judge={jv:18s} human={hv:18s} : {c}")
    lines.append("=" * 60)
    return "\n".join(lines)


def print_report(store: Store, judge_id: str, judge_version: str,
                 labeler_id: str, second_labeler: str | None = None) -> None:
    jr = judge_vs_human(store, judge_id, judge_version, labeler_id)
    ceiling = human_vs_human(store, labeler_id, second_labeler) if second_labeler else None
    print(format_report(jr, judge_label=f"{judge_id}@{judge_version[:8]}", ceiling=ceiling))


def validate_report(store, judge_id, judge_version, labeler_id) -> str:
    """Ambiguous-aware report: kappa on decidable cases + separate
    reporting of how the judge handled human-ambiguous cases."""
    judge_map = {k: v for k, v in
                 store.judge_verdicts(judge_id, judge_version).items()
                 if v != PARSE_ERROR}
    human_map = store.latest_labels(labeler_id)
    ls_j, ls_h = store.build_label_sets(
        (judge_id, judge_map), (labeler_id, human_map),
        kind_a="judge", kind_b="human")
    vr = validate(ls_j.sample_ids, ls_j.verdicts, ls_h.verdicts)

    lines = ["=" * 60,
             f"JUDGE RELIABILITY (ambiguous-aware) — {judge_id}",
             "=" * 60,
             f"  DECIDABLE cases (human gave firm verdict): {vr.decidable.n}",
             f"    raw agreement : {vr.decidable.raw_agreement:.2f}",
             f"    Cohen's kappa : {vr.decidable.cohens_kappa:.2f}",
             "-" * 60,
             f"  AMBIGUOUS cases (human unsure): {vr.n_ambiguous}",
             "    (judge cannot be 'wrong' here — no ground truth)"]
    if vr.n_ambiguous:
        lines.append("    How the judge handled them:")
        for verdict, count in sorted(vr.judge_on_ambiguous.items(), key=lambda x: -x[1]):
            lines.append(f"      judge said {verdict:18s}: {count}")
        forced = sum(c for v, c in vr.judge_on_ambiguous.items() if v != "ambiguous")
        lines.append(f"    → judge forced a firm verdict on {forced}/"
                     f"{vr.n_ambiguous} ambiguous cases (calibration gap)")
    lines.append("-" * 60)
    k = vr.decidable.cohens_kappa
    if k >= 0.8:
        lines.append("  VERDICT: Strong on decidable cases — trustworthy there.")
    elif k >= 0.6:
        lines.append("  VERDICT: Moderate on decidable cases — verify edges.")
    else:
        lines.append("  VERDICT: Weak even on decidable cases — do not trust.")
    lines.append("=" * 60)
    return "\n".join(lines)
