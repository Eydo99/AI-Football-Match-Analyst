"""
Two strategies, meant to be used together:

  subsample_by_match   - keep a random fraction of match_ids. Preserves the
                         match-grouped structure that GroupShuffleSplit and
                         StratifiedGroupKFold depend on. Slicing random rows
                         instead would split single matches across the
                         train/test boundary, which is the leak the whole
                         grouped-CV design exists to prevent.

  cap_majority_classes - cap Pass/Carry (81% of rows between them) at a fixed
                         row count while keeping every Shot and Foul row.
                         Shrinks the data and raises the rare-class share at
                         the same time.
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


def cap_majority_classes(X, y, groups, cap=150_000,
                         majority_classes=MAJORITY_CLASSES, random_state=42):

    rng = np.random.RandomState(random_state)
    keep_idx = []

    for class_label in y.unique():
        class_idx = y.index[y == class_label]
        if class_label in majority_classes and len(class_idx) > cap:
            class_idx = rng.choice(class_idx, size=cap, replace=False)
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