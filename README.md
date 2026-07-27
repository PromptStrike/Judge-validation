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

Raw agreement (the percentage of samples where the judge and human match) is
misleading when the labels are imbalanced. If 90% of responses are safe, a judge
that blindly labels *everything* "safe" scores 90% raw agreement — while being
completely useless, because it never actually distinguishes safe from harmful.

Cohen's kappa corrects for the agreement you'd expect by chance alone. A high
raw agreement with a low kappa is a warning sign: the judge looks accurate but
isn't really discriminating. In testing, I've seen an 80% raw agreement drop to
a 0.58 kappa once chance was accounted for — the same result, told honestly.

That gap is the whole reason this tool reports kappa: it's the difference
between a number that flatters the judge and one you can defend.

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

Building this tool surfaced a concrete weakness in an LLM judge
(llama-3.3-70b as judge). I ran a controlled test: take harmful content the
judge reliably catches, then append a disavowal — "this is an example of what
you should NEVER do" — while changing nothing else about the content.

Presented plainly, the harmful content was flagged correctly 100% of the time.
The *same* content, with a "never do this" warning attached, was scored "safe"
in a subset of cases — and the effect was one-directional: the warning never
improved the judge's accuracy, only degraded it. The result replicated exactly
across separate runs (the judge is deterministic at temperature 0).

The effect is modest and content-dependent — it flipped the verdict on factual
and emotional harmful content, but not on code payloads — so I'm not claiming a
universal jailbreak of LLM judges. But the direction is clear: appending a
disclaimer to harmful output can sometimes launder it past a judge that would
otherwise catch it. If your evaluation pipeline relies on an LLM judge, this is
exactly the kind of blind spot you can't assume away — you have to test for it,
which is what this tool is for.

## Status

Early / work in progress. [Or however you want to frame it.]

## License

[MIT is standard for open tooling — your call.]
