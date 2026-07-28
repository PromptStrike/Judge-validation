"""Validation sampling — select a subset to hand-label from a large dataset.

You never label the whole dataset; you label a representative sample to
validate the judge, then trust the judge on the rest (if it passes).

Two strategies:
  random     — uniform random N samples
  stratified — N samples spread across the judge's verdict categories, so
               the validation set covers cases the judge was confident AND
               unsure about (avoids validating only on easy cases)
"""
from __future__ import annotations

import random as _random
from collections import defaultdict

from jval.models import Sample
from jval.store import Store


def sample_random(store: Store, n: int, *,
                  labeler_id: str | None = None,
                  seed: int | None = None) -> list[Sample]:
    """N uniformly random samples. If labeler_id given, only unlabeled ones."""
    pool = store.samples()
    if labeler_id is not None:
        done = store.labeled_sample_ids(labeler_id)
        pool = [s for s in pool if s.id not in done]
    if n >= len(pool):
        return pool
    return _random.Random(seed).sample(pool, n)


def sample_stratified(store: Store, n: int, judge_id: str, judge_version: str, *,
                      labeler_id: str | None = None,
                      seed: int | None = None) -> list[Sample]:
    """N samples spread proportionally across the judge's verdict categories.

    Ensures the validation set includes cases from each thing the judge said,
    so you don't accidentally validate only on the judge's confident cases.
    Falls back to random for samples the judge hasn't scored.
    """
    verdicts = store.judge_verdicts(judge_id, judge_version)   # {sample_id: verdict}
    pool = store.samples()
    if labeler_id is not None:
        done = store.labeled_sample_ids(labeler_id)
        pool = [s for s in pool if s.id not in done]

    # group samples by the judge's verdict
    by_verdict: dict[str, list[Sample]] = defaultdict(list)
    for s in pool:
        by_verdict[verdicts.get(s.id, "UNSCORED")].append(s)

    rng = _random.Random(seed)
    total = len(pool)
    if n >= total:
        return pool

    picked: list[Sample] = []
    # proportional allocation per stratum
    for verdict, group in by_verdict.items():
        share = max(1, round(n * len(group) / total))
        share = min(share, len(group))
        picked.extend(rng.sample(group, share))

    # trim or top up to exactly n (rounding can over/undershoot)
    rng.shuffle(picked)
    if len(picked) > n:
        picked = picked[:n]
    elif len(picked) < n:
        remaining = [s for s in pool if s not in picked]
        picked.extend(rng.sample(remaining, min(n - len(picked), len(remaining))))
    return picked


def recommended_sample_size(dataset_size: int) -> int:
    """A sane validation sample size for a given dataset.

    Not statistically exhaustive — a practical heuristic:
      - below ~50: label everything
      - otherwise: ~150 is enough for a stable kappa; more has diminishing returns
    """
    if dataset_size <= 50:
        return dataset_size
    return min(150, dataset_size)
