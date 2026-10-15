"""Report how often each splitter fallback fires on a random sample of real trials.

    python scripts/criteria_fallback_stats.py [--sample 500] [--seed 1]

Streams the whole result set from the API (about 20 requests), keeps a uniform random sample
(reservoir sampling) and prints counts and percentages.
"""

import argparse
import random

from trial_matcher.ctgov_client import CTGovClient
from trial_matcher.split_criteria import split_criteria, summarize_notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sample: list[str] = []
    seen = 0
    for study in CTGovClient().iter_studies():
        text = study["protocolSection"].get("eligibilityModule", {}).get("eligibilityCriteria")
        seen += 1
        if len(sample) < args.sample:
            sample.append(text)
        elif (j := rng.randrange(seen)) < args.sample:
            sample[j] = text

    results = [split_criteria(t) for t in sample]
    counts = summarize_notes(results)
    n = counts.pop("trials")
    print(f"sampled {n} of {seen} trials (seed {args.seed})")
    for note, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {note:28s} {c:4d}  {100 * c / n:5.1f}%")
    sizes = sorted(len(r.criteria) for r in results)
    print(f"criteria per trial: min {sizes[0]}, median {sizes[n // 2]}, max {sizes[-1]}")
    print(f"trials with zero criteria: {sum(1 for s in sizes if s == 0)}")


if __name__ == "__main__":
    main()
