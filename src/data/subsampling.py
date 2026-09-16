"""
Two different tools for two different jobs — do not combine them:

  subsample_by_match   - keep a random fraction of match_ids. Shrinks row
                         count WITHOUT changing class balance: every class's
                         real-world share is preserved (in expectation),
                         since no class is targeted for extra reduction.
                         Safe to apply to the full dataset before splitting,
                         including the test portion, because it's still an
                         unbiased sample of the real population — just a
                         smaller, noisier one. THIS is the default choice
                         for "same proportions, fewer rows."

  cap_majority_classes - deliberately REBALANCES classes (e.g. caps Pass and
                         Carry while leaving Shot/Foul untouched), which
                         changes what the data represents. Only use this on
                         the training split, after it's separated from test
                         — applying it to data that includes your test set
                         corrupts reported metrics, since you'd be measuring
                         performance against a population that doesn't exist
                         in real matches. Reach for this only if you
                         specifically want to fight class imbalance during
                         training, not as a row-count reduction tool.

check_min_class_count - run after subsample_by_match to catch the case where
                         a rare class (Shot at ~1.1%) has shrunk to a count
                         too small for stable per-fold macro-F1 estimates.
"""
import numpy as np
import pandas as pd

MAJORITY_CLASSES = ('Pass', 'Carry')


def subsample_by_match(X, y, groups, match_frac=0.2, random_state=42):

    if not 0 < match_frac <= 1:
        raise ValueError("match_frac must be in (0, 1]")

    unique_matches = pd.Series(groups.unique())
    n_keep = max(1, int(round(len(unique_matches) * match_frac)))
    rng = np.random.RandomState(random_state)
    keep = set(rng.choice(unique_matches, size=n_keep, replace=False))

    mask = groups.isin(keep)
    return X[mask], y[mask], groups[mask]


def check_min_class_count(y, n_splits, min_per_fold=30):
    counts = y.value_counts()
    per_fold = counts / n_splits
    risky = per_fold[per_fold < min_per_fold]
    if len(risky):
        print("WARNING: these classes may be too small for stable CV at "
              f"n_splits={n_splits} (fewer than {min_per_fold} rows/fold expected):")
        for class_label, n in risky.items():
            print(f"  {class_label}: ~{n:.0f} rows/fold (total {counts[class_label]:,})")
    return risky


def cap_majority_classes(X, y, groups, cap, majority_classes=MAJORITY_CLASSES, random_state=42):
    rng = np.random.RandomState(random_state)

    if isinstance(cap, dict):
        caps = cap
    else:
        caps = {class_label: cap for class_label in majority_classes}

    keep_idx = []
    for class_label in y.unique():
        class_idx = y.index[y == class_label]
        class_cap = caps.get(class_label)
        if class_cap is not None and len(class_idx) > class_cap:
            class_idx = rng.choice(class_idx, size=class_cap, replace=False)
        keep_idx.append(pd.Index(class_idx))

    keep = pd.Index(np.concatenate([idx.values for idx in keep_idx]))
    keep = keep.sort_values()
    return X.loc[keep], y.loc[keep], groups.loc[keep]


def describe_balance(y, label=''):
    """Print class counts and percentages — handy before/after subsampling."""
    counts = y.value_counts()
    pct = (counts / len(y) * 100).round(1)
    print(f"\n{label} ({len(y):,} rows)")
    for class_label in counts.index:
        print(f"  {class_label:<8} {counts[class_label]:>9,}  {pct[class_label]:>5}%")
    return counts