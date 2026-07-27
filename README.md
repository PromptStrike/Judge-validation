# jval — Judge Validation Harness

A tool for measuring whether an LLM "judge" can be trusted.

## The problem

LLM evaluations increasingly rely on **LLM-as-judge**: using one model to
score another model's outputs (did the attack succeed? did the model refuse?
is this response harmful?). At scale, humans can't check every output, so an
AI judge does it automatically.

The catch: **these judges are often wrong, and almost nobody checks.** A judge
that scores confidently but inaccurately produces confidently-wrong metrics —
and every decision built on those metrics inherits the error.

`jval` makes judge validation a repeatable step instead of a one-off manual
chore: hand-label a sample, run the judge over the same data, and get a
chance-corrected reliability report that shows not just *how often* the judge
is wrong, but *how* it's wrong.

## What it does

- Load model outputs from any source (attack tools, logs, benchmarks) as samples
- Capture human ground truth via **blind labeling** (you don't see the judge's
  verdict while labeling, so your labels stay independent)
- Run an LLM judge over the samples
- Report agreement using **Cohen's kappa** (chance-corrected, not raw agreement)
  plus a confusion breakdown showing the *direction* of disagreements

## Why kappa, not accuracy

[Explain in your own words: raw agreement lies when classes are imbalanced —
a judge that always says "safe" scores high accuracy on a mostly-safe set while
being useless. Kappa corrects for chance. Reference your own experience if you
want.]

## Install

```bash
git clone <repo-url>
cd judge-validation
python3 -m venv .venv
.venv/bin/pip install -e .
export GROQ_API_KEY="your-key"
```

## Usage

[A minimal example — loading samples, labeling, running the judge, printing the
report. You can adapt one of your working scripts (dry_run.py / api_run.py).]

## Architecture

- `models.py` — data structures (Sample, HumanLabel, JudgeVerdict)
- `store.py` — JSONL persistence
- `labeling.py` — blind human labeling session
- `judges/` — the judge interface + LLM judge
- `runner.py` — runs a judge over all samples (resumable)
- `analysis.py` — agreement metrics (kappa, confusion matrix)
- `report.py` — reliability report

## A finding

[This is the strong part — briefly describe your warning-test result:
controlled test, plain harmful content caught 100%, same content with
"never do this" warnings fooled the judge in some cases, one-directional
effect. This turns the README from "here's a tool" into "here's a tool AND
something real I found with it." Keep it honest about the modest effect size.]

## Status

Early / work in progress. [Or however you want to frame it.]

## License

[MIT is standard for open tooling — your call.]
