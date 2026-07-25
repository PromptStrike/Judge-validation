"""Judge interface. Any judge (LLM, heuristic, human-proxy) implements this,
so the runner and analysis never care which judge they're using.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jval.models import Sample, JudgeSpec, JudgeVerdict


@dataclass(frozen=True)
class JudgeOutput:
    """What a judge returns for one sample, before storage."""
    verdict: str          # normalized label, or "PARSE_ERROR"
    reasoning: str        # why — for auditing
    raw: str              # the untouched model output


class Judge(ABC):
    spec: JudgeSpec

    @abstractmethod
    def judge(self, sample: Sample) -> JudgeOutput:
        ...

    def to_verdict(self, sample: Sample, out: JudgeOutput) -> JudgeVerdict:
        """Wrap a JudgeOutput into a storable, version-stamped JudgeVerdict."""
        return JudgeVerdict(
            sample_id=sample.id,
            judge_id=self.spec.judge_id,
            judge_version=self.spec.version,
            verdict=out.verdict,
            raw_output=out.raw,
        )
