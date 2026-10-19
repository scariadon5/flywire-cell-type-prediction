"""Seed / validation / test splits for every task, saved once so every model sees identical splits.

Two settings:
  sweep       label a fraction of neurons at random, guaranteeing at least one seed per class
              (NTAC's "at_least_one_per_class" protocol); 5 repetitions per fraction
  hemisphere  seeds are every labelled neuron on one side; predict the other side

Of the non-seed labelled neurons, VAL_SHARE go to validation (for tuning and early stopping) and the
rest to test. Neurons without a label at the task's level stay in the graph with role NONE: they are
never seeds and never scored.

Output: data/splits/<task>.parquet, one row per proofread neuron, `root_id` plus one int8 role column
per split (e.g. `frac05_r2`, `hemi_l2r`). data/splits_report.json records each split's sizes and an md5
of its roles; it is committed, so a rebuild anywhere (e.g. Kaggle) is checked against it.
"""

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from flyct import config

TASKS = ["super_class", "cell_class", "cell_type"]
FRACTIONS = [0.01, 0.05, 0.20, 0.80]
REPS = 5
VAL_SHARE = 0.10
NONE, SEED, VAL, TEST = 0, 1, 2, 3
HEMISPHERES = {"hemi_l2r": ("left", "right"), "hemi_r2l": ("right", "left")}


def sweep_name(frac: float, rep: int) -> str:
    return f"frac{round(frac * 100):02d}_r{rep}"


def split_names() -> list[str]:
    return [sweep_name(f, r) for f in FRACTIONS for r in range(REPS)] + list(HEMISPHERES)


def _assign_rest(roles, pool, rng):
    """Shuffle the non-seed labelled neurons into validation and test."""
    pool = rng.permutation(pool)
    n_val = int(round(VAL_SHARE * len(pool)))
    roles[pool[:n_val]] = VAL
    roles[pool[n_val:]] = TEST


def sweep_split(y: np.ndarray, frac: float, rng) -> np.ndarray:
    labelled = np.flatnonzero(pd.notna(y))
    roles = np.full(len(y), NONE, dtype=np.int8)
    # One random seed per class first, then fill uniformly up to the target count.
    per_class = pd.Series(labelled).groupby(y[labelled]).apply(lambda s: rng.choice(s.values))
    seeds = set(per_class.values.tolist())
    target = int(frac * len(labelled))
    if target > len(seeds):
        rest = np.setdiff1d(labelled, np.fromiter(seeds, dtype=np.int64))
        seeds.update(rng.choice(rest, target - len(seeds), replace=False).tolist())
    seeds = np.fromiter(seeds, dtype=np.int64)
    roles[seeds] = SEED
    _assign_rest(roles, np.setdiff1d(labelled, seeds), rng)
    return roles


def hemisphere_split(y: np.ndarray, side: np.ndarray, src: str, dst: str, rng) -> np.ndarray:
    labelled = pd.notna(y)
    roles = np.full(len(y), NONE, dtype=np.int8)
    roles[labelled & (side == src)] = SEED
    _assign_rest(roles, np.flatnonzero(labelled & (side == dst)), rng)
    return roles


def make_splits(labels: pd.DataFrame, task: str) -> pd.DataFrame:
    y = labels[task].to_numpy(dtype=object)
    side = labels["side"].to_numpy(dtype=object)
    out = {"root_id": labels["root_id"].to_numpy()}
    t = TASKS.index(task)
    for frac in FRACTIONS:
        for rep in range(REPS):
            rng = np.random.default_rng([t, round(frac * 100), rep])
            out[sweep_name(frac, rep)] = sweep_split(y, frac, rng)
    for i, (name, (src, dst)) in enumerate(HEMISPHERES.items()):
        out[name] = hemisphere_split(y, side, src, dst, np.random.default_rng([t, 1000 + i]))
    return pd.DataFrame(out)


def load(task: str, split: str) -> pd.Series:
    """Roles for one split, indexed by root_id (same order as features.parquet and labels.parquet)."""
    df = pd.read_parquet(config.DATA / "splits" / f"{task}.parquet", columns=["root_id", split])
    return df.set_index("root_id")[split]


def main() -> int:
    labels = pd.read_parquet(config.PROCESSED / "labels.parquet")
    out_dir = config.DATA / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for task in TASKS:
        df = make_splits(labels, task)
        y = labels[task].to_numpy(dtype=object)
        n_classes = labels[task].nunique()
        for name in split_names():
            r = df[name].to_numpy()
            seeds = r == SEED
            assert not (pd.isna(y) & (r != NONE)).any(), f"{task}/{name}: unlabelled neuron given a role"
            if name.startswith("frac"):
                assert pd.Series(y[seeds]).nunique() == n_classes, f"{task}/{name}: a class has no seed"
            report[f"{task}/{name}"] = {
                "seed": int(seeds.sum()), "val": int((r == VAL).sum()), "test": int((r == TEST).sum()),
                "seed_classes": int(pd.Series(y[seeds]).nunique()),
                "md5": hashlib.md5(r.tobytes()).hexdigest(),  # the exact roles, in root_id order
            }
        df.to_parquet(out_dir / f"{task}.parquet", index=False)
        print(f"{task}: {n_classes:,} classes, {len(split_names())} splits")

    path = config.DATA / "splits_report.json"
    if path.exists() and json.loads(path.read_text()) != report:
        print(f"splits differ from the committed {path.name}; delete it to accept the new splits", file=sys.stderr)
        return 1
    path.write_text(json.dumps(report, indent=1) + "\n")
    for key in ("cell_type/frac01_r0", "cell_type/frac80_r0", "cell_type/hemi_l2r", "super_class/frac01_r0"):
        print(f"  {key}: {report[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
