"""Core data structures. No logic, no I/O — just shapes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Sample:
    id: str
    prompt: str
    response: str
    source_meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HumanLabel:
    sample_id: str
    labeler_id: str
    verdict: str
    notes: str = ""
    labeled_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class JudgeSpec:
    judge_id: str
    model: str
    rubric: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        payload = json.dumps(
            {"model": self.model, "rubric": self.rubric, "params": self.params},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class JudgeVerdict:
    sample_id: str
    judge_id: str
    judge_version: str
    verdict: str
    raw_output: str = ""
    judged_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class LabelSet:
    rater_id: str
    rater_kind: str
    sample_ids: list[str]
    verdicts: list[str]

    def __post_init__(self) -> None:
        if len(self.sample_ids) != len(self.verdicts):
            raise ValueError("sample_ids and verdicts must align")


def to_dict(obj: Any) -> dict:
    return asdict(obj)
