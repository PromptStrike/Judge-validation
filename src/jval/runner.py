"""Runner — executes a judge over stored samples and persists verdicts.

Handles the practical realities of scoring at scale: skip already-judged
samples (resumable), retry transient failures, track parse errors, and
report progress. This is the automation layer over a single judge call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from jval.store import Store
from jval.judges.base import Judge
from jval.judges.llm_judge import PARSE_ERROR


@dataclass
class RunSummary:
    total: int = 0
    judged: int = 0
    skipped: int = 0          # already had a verdict for this judge version
    parse_errors: int = 0
    api_retries: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"judged={self.judged} skipped={self.skipped} "
                f"parse_errors={self.parse_errors} retries={self.api_retries} "
                f"(total={self.total})")


def run_judge(store: Store,
              judge: Judge,
              *,
              resume: bool = True,
              max_retries: int = 2,
              retry_delay: float = 2.0,
              progress: bool = True) -> RunSummary:
    """Score every sample in the store with `judge`, saving each verdict.

    resume=True  : skip samples this judge VERSION already scored (so a
                   re-run after a crash continues rather than re-billing).
    max_retries  : transient API failures are retried; a persistent failure
                   is recorded as PARSE_ERROR, never silently dropped.
    """
    summary = RunSummary()
    samples = store.samples()
    summary.total = len(samples)

    # which sample_ids this exact judge version already has verdicts for
    already = set()
    if resume:
        already = set(store.judge_verdicts(judge.spec.judge_id,
                                           judge.spec.version))

    for i, sample in enumerate(samples, 1):
        if sample.id in already:
            summary.skipped += 1
            continue

        out = None
        for attempt in range(max_retries + 1):
            out = judge.judge(sample)
            if out.verdict != PARSE_ERROR:
                break
            # PARSE_ERROR might be a transient API blip — retry a couple times
            if attempt < max_retries:
                summary.api_retries += 1
                time.sleep(retry_delay)

        verdict = judge.to_verdict(sample, out)
        store.add_verdict(verdict)
        summary.judged += 1
        if out.verdict == PARSE_ERROR:
            summary.parse_errors += 1
            summary.errors.append(f"{sample.id}: {out.reasoning}")

        if progress and (i % 10 == 0 or i == summary.total):
            print(f"  [{i}/{summary.total}] {summary}")

    return summary
