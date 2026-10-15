# Retrieval eval

Measures how well trial search finds the right trials for a patient profile.
All profiles are synthetic (no real patient data).

| File | What it is |
|---|---|
| `profiles.json` | 30 synthetic patient profiles: text, age, sex, tags, and a short "relevant if / not relevant if" note |
| `labels.csv` | Graded trials per profile. This is the file a human reviews and edits |
| `results/` | Saved eval reports. Each one says which model, text variant and labels it used |

## Grades

| Grade | Meaning |
|---|---|
| **2** | Right disease, and nothing stated in the trial excludes this patient. Worth screening. |
| **1** | Right disease, but fit is uncertain: the excerpt is silent on line of therapy or biomarker, it is a basket or platform trial, or a key detail cannot be told. |
| **0** | Wrong disease, **or** a stated requirement clearly excludes the patient (first-line only but the patient is pretreated, needs a biomarker the patient lacks, wrong stage or setting, wrong population). |

Wrong line of therapy is a 0, not a 1. That is deliberate: it is the nuance plain vector search ignores.

## How labels were made

1. For every profile, the top 10 trials of each method (full-text, vector, hybrid) were pooled. Duplicates were merged. That gave 556 rows.
2. Each row was graded 0/1/2 by Claude from the title, conditions, and an excerpt of the criteria (the first inclusion criteria plus any that mention prior treatment, line, or stage). These are **suggestions**, kept untouched in `suggested_grade`.
3. A human reviews `grade`, profile by profile.

## How to review

Open `labels.csv` in a spreadsheet (Numbers, Excel, Google Sheets) or a text editor. Rows are grouped by profile.

- Read the `profile` column, then each trial's `title`, `conditions`, and `key_criteria` (I: inclusion, E: exclusion). `url` opens the full ClinicalTrials.gov record. `reason` is the first-pass reason.
- If the grade is wrong, **change `grade` only** (0, 1 or 2).
- When you have gone through **every row of a profile**, put `y` in `reviewed` on all of that profile's rows. A profile only counts as reviewed when all of its rows are marked.
- Do not edit `suggested_grade`, `profile`, or the other display columns. They let us measure how often the suggestions were changed.
- Save as CSV (UTF-8). Do not add or remove columns.
- Check progress at any time: `python -m trial_matcher.evaluate status`

## Known limits

- **The pool is shallow (top 10 per method).** A relevant trial that every method ranks 11th or lower is never judged, so it counts as not relevant. Scores are therefore a lower bound. P26 and P27 have no grade-2 trial in the pool even though relevant trials exist deeper in the search results, and are skipped in the strict metrics. `pool --depth 20` adds about 530 more rows for review.
- **recall@50 is flattering.** Each method's top 10 is inside the pool by construction, so a method always recovers its own pooled trials. Compare methods with each other, not with an absolute target.
- `judged@10` shows how much of a method's top 10 has a label. It is 1.00 for fts, vector and hybrid on the current labels; a new model or text variant will score lower until its new candidates are pooled (`pool --model ... --variant ...`) and graded.
- Labels reflect the 19,455-trial snapshot of 2026-10-15. Profiles are tuned to oncology.

## Metrics

Relevant means grade >= 2 (`--min-grade 1` makes it lenient). Profiles with no relevant trial are skipped.

- **recall@50**: share of the profile's relevant trials found in the top 50.
- **nDCG@10**: ranking quality in the top 10, gain = grade (0/1/2), log2 discount, 1.0 is a perfect order. Unlabelled results count as 0.
- **hit@10**: 1 if at least one relevant trial is in the top 10.

All three are averaged over profiles. The report also breaks them out by profile tag (for example `prior_treatment`).

## Commands

```bash
python -m trial_matcher.evaluate status                        # review progress (no database needed)
python -m trial_matcher.evaluate run --out evals/results/NAME.txt   # score fts / vector / hybrid
python -m trial_matcher.evaluate run --only-reviewed           # score only fully reviewed profiles
python -m trial_matcher.evaluate pool --depth 20               # add more candidates to review
```

Every change to retrieval or to an LLM prompt or model must be measured here before and after, and the numbers go in the commit or PR description.
