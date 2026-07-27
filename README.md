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
  plus a confusion breakdown showing the *direction* of
