"""Run models on saved splits, score them, and log every run to MLflow and to results/runs.jsonl.

    python -m flyct.run --method majority lightgbm --task super_class cell_class --split frac hemi

`--split` takes split names or prefixes: `frac05` runs frac05_r0..r4, `frac` every sweep split,
`hemi` both hemisphere directions, `all` everything.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from flyct import config, features, metrics, splits
from flyct.models import MODELS

RESULTS = config.ROOT / "results" / "runs.jsonl"


@dataclass
class TaskData:
    task: str
    root_id: np.ndarray  # (n,) all proofread neurons
    y: np.ndarray        # (n,) labels at the task's level, None where unlabelled
    roles: np.ndarray    # (n,) NONE / SEED / VAL / TEST
    X: np.ndarray        # (n, f) features
    feature_names: list

    @property
    def seed(self):
        return np.flatnonzero(self.roles == splits.SEED)

    @property
    def val(self):
        return np.flatnonzero(self.roles == splits.VAL)

    @property
    def test(self):
        return np.flatnonzero(self.roles == splits.TEST)

    @property
    def scored(self):
        """Neurons a model must predict: validation then test, each in root_id order."""
        return np.concatenate([self.val, self.test])


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=config.ROOT, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, cwd=config.ROOT).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def expand_splits(names: list[str]) -> list[str]:
    every = splits.split_names()
    if "all" in names:
        return every
    out = [s for s in every if any(s == n or s.startswith(n) for n in names)]
    if not out:
        sys.exit(f"no split matches {names}; choose from {every}")
    return out


def split_params(split: str) -> dict:
    if split.startswith("frac"):
        frac, rep = split[4:].split("_r")
        return {"setting": "sweep", "fraction": int(frac) / 100, "rep": int(rep)}
    return {"setting": split, "fraction": None, "rep": 0}


def score(data: TaskData, idx, topk, conf, region, class_size) -> dict:
    return metrics.evaluate(data.y[idx], topk, conf, region=region[idx], class_size=class_size[idx])


def run_one(method, data, split, region, class_size, tracker):
    t0 = time.perf_counter()
    topk, conf = MODELS[method](data)
    seconds = time.perf_counter() - t0
    n_val = len(data.val)
    m_val = score(data, data.val, topk[:n_val], None if conf is None else conf[:n_val], region, class_size)
    m_test = score(data, data.test, topk[n_val:], None if conf is None else conf[n_val:], region, class_size)
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": method, "task": data.task, "split": split, **split_params(split),
        "git_commit": git_commit(), "annotations": config.ANNOTATIONS_TAG,
        "n_seed": len(data.seed), "seconds": round(seconds, 2),
        **{f"val_{k}": v for k, v in m_val.items() if k in ("acc", "top5_acc", "f1_macro")},
        **{f"test_{k}": v for k, v in m_test.items()},
    }
    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")
    tracker(record)
    print(f"{method:>9} {data.task:>11} {split:>10}  test acc {m_test['acc']:.3f}"
          f"  macro F1 {m_test['f1_macro']:.3f}  ({seconds:.1f}s)")


def mlflow_tracker():
    try:
        import mlflow
    except ImportError:
        return lambda record: None
    mlflow.set_tracking_uri(f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("flyct")

    def log(record):
        params = {k: v for k, v in record.items() if not isinstance(v, float) or k == "fraction"}
        with mlflow.start_run(run_name=f"{record['method']}/{record['task']}/{record['split']}"):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in record.items() if k not in params})
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--task", nargs="+", required=True, choices=splits.TASKS)
    ap.add_argument("--split", nargs="+", default=["all"])
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    labels = pd.read_parquet(config.PROCESSED / "labels.parquet")
    feats = features.load()
    assert (feats.index.to_numpy() == labels.root_id.to_numpy()).all(), "features and labels out of order"
    X = feats.to_numpy()
    region = labels.super_class.fillna("none").to_numpy(dtype=object)
    tracker = (lambda record: None) if args.no_mlflow else mlflow_tracker()

    for task in args.task:
        y = labels[task].to_numpy(dtype=object)
        class_size = labels[task].map(labels[task].value_counts()).fillna(0).to_numpy()
        roles_all = pd.read_parquet(config.DATA / "splits" / f"{task}.parquet")
        assert (roles_all.root_id.to_numpy() == labels.root_id.to_numpy()).all()
        for split in expand_splits(args.split):
            data = TaskData(task, labels.root_id.to_numpy(), y, roles_all[split].to_numpy(), X, list(feats.columns))
            for method in args.method:
                run_one(method, data, split, region, class_size, tracker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
