"""Blind labeling session.

Enforces blindness STRUCTURALLY: this module imports Sample and HumanLabel
but NOT JudgeVerdict, and the Store methods it calls never return verdicts.
It is therefore impossible for a labeler using this session to see what a
judge decided. The guarantee is architectural, not a matter of remembering.

No network, no API — pure local labeling.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from jval.models import Sample, HumanLabel
from jval.store import Store


@dataclass
class LabelingSession:
    store: Store
    labeler_id: str
    allowed_verdicts: tuple[str, ...] = ("success", "not_success", "ambiguous")
    shuffle: bool = True
    seed: int | None = None

    def _pending(self) -> list[Sample]:
        """Samples this labeler hasn't labeled yet."""
        done = self.store.labeled_sample_ids(self.labeler_id)
        pending = [s for s in self.store.samples() if s.id not in done]
        if self.shuffle:
            random.Random(self.seed).shuffle(pending)   # break order bias
        return pending

    def record(self, sample_id: str, verdict: str, notes: str = "") -> HumanLabel:
        """Record one verdict. Validates against the allowed set."""
        if verdict not in self.allowed_verdicts:
            raise ValueError(
                f"verdict {verdict!r} not in {self.allowed_verdicts}")
        label = HumanLabel(sample_id=sample_id, labeler_id=self.labeler_id,
                           verdict=verdict, notes=notes)
        self.store.add_label(label)
        return label

    def run_cli(self, prompt_fn: Callable[[Sample], str] | None = None) -> int:
        """Interactive terminal labeling. Returns count labeled this run.

        Shows ONLY prompt + response. No scores, no judge output — there's
        nothing here that could leak a verdict, by construction.
        """
        pending = self._pending()
        if not pending:
            print("Nothing to label — all samples done for", self.labeler_id)
            return 0

        print(f"\n{len(pending)} samples to label as '{self.labeler_id}'")
        print(f"Verdicts: {', '.join(self.allowed_verdicts)} "
              f"| 's' skip | 'q' quit\n")

        n = 0
        for i, s in enumerate(pending, 1):
            print("=" * 70)
            print(f"[{i}/{len(pending)}]  sample_id: {s.id}")
            print("-" * 70)
            print("PROMPT:\n", s.prompt)
            print("-" * 70)
            print("RESPONSE:\n", s.response)
            print("=" * 70)

            while True:
                choice = input("verdict> ").strip().lower()
                if choice == "q":
                    print(f"\nStopped. Labeled {n} this session.")
                    return n
                if choice == "s":
                    break
                # allow prefix match: 'su' -> 'success'
                matches = [v for v in self.allowed_verdicts if v.startswith(choice)]
                if len(matches) == 1:
                    note = input("note (optional)> ").strip()
                    self.record(s.id, matches[0], note)
                    n += 1
                    break
                print(f"  ? enter one of {self.allowed_verdicts}, 's', or 'q'")
        print(f"\nDone. Labeled {n} this session.")
        return n
