"""Persistence layer. JSONL, append-only for labels. I/O only, no logic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from jval.models import Sample, HumanLabel, JudgeVerdict, LabelSet, to_dict


class Store:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.root / "samples.jsonl"
        self.labels_path = self.root / "human_labels.jsonl"
        self.verdicts_path = self.root / "judge_verdicts.jsonl"

    @staticmethod
    def _append(path: Path, obj) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_dict(obj), ensure_ascii=False) + "\n")

    @staticmethod
    def _read(path: Path) -> Iterator[dict]:
        if not path.exists():
            return iter(())
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # samples
    def add_sample(self, sample: Sample) -> None:
        self._append(self.samples_path, sample)

    def add_samples(self, samples: list[Sample]) -> None:
        for s in samples:
            self.add_sample(s)

    def samples(self) -> list[Sample]:
        return [Sample(**d) for d in self._read(self.samples_path)]

    # human labels (append-only)
    def add_label(self, label: HumanLabel) -> None:
        self._append(self.labels_path, label)

    def labels(self) -> list[HumanLabel]:
        return [HumanLabel(**d) for d in self._read(self.labels_path)]

    def latest_labels(self, labeler_id: str) -> dict[str, str]:
        out: dict[str, tuple[str, str]] = {}
        for lab in self.labels():
            if lab.labeler_id != labeler_id:
                continue
            prev = out.get(lab.sample_id)
            if prev is None or lab.labeled_at > prev[0]:
                out[lab.sample_id] = (lab.labeled_at, lab.verdict)
        return {sid: v for sid, (_, v) in out.items()}

    def labeled_sample_ids(self, labeler_id: str) -> set[str]:
        return set(self.latest_labels(labeler_id))

    # judge verdicts
    def add_verdict(self, verdict: JudgeVerdict) -> None:
        self._append(self.verdicts_path, verdict)

    def verdicts(self) -> list[JudgeVerdict]:
        return [JudgeVerdict(**d) for d in self._read(self.verdicts_path)]

    def judge_verdicts(self, judge_id: str, judge_version: str) -> dict[str, str]:
        return {
            v.sample_id: v.verdict
            for v in self.verdicts()
            if v.judge_id == judge_id and v.judge_version == judge_version
        }

    def build_label_sets(
        self,
        rater_a: tuple[str, dict[str, str]],
        rater_b: tuple[str, dict[str, str]],
        kind_a: str = "judge",
        kind_b: str = "human",
    ) -> tuple[LabelSet, LabelSet]:
        (id_a, map_a), (id_b, map_b) = rater_a, rater_b
        common = sorted(set(map_a) & set(map_b))
        if not common:
            raise ValueError("no overlapping samples between the two raters")
        return (
            LabelSet(id_a, kind_a, common, [map_a[s] for s in common]),
            LabelSet(id_b, kind_b, common, [map_b[s] for s in common]),
        )
