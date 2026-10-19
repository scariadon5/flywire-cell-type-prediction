"""Check flyct.ntac against the published NTAC package and the paper, on NTAC's own FlyWire subsets.

Needs the reference package, which is not a project dependency. Run it from a separate environment:

    py -3.12 -m venv .venv-ntac
    .venv-ntac/Scripts/python -m pip install git+https://github.com/BenJourdan/ntac@04a8e46 -e .
    .venv-ntac/Scripts/python scripts/ntac_reproduction.py fetch       # their data, 134 MB
    NUMBA_NUM_THREADS=1 .venv-ntac/Scripts/python scripts/ntac_reproduction.py agree
    .venv-ntac/Scripts/python scripts/ntac_reproduction.py paper

agree  runs both implementations on one split and reports how often they make the same prediction.
       Run it single-threaded: the reference similarity kernel adds into a shared matrix inside a
       parallel loop over dimensions, which races, so its multi-threaded output varies run to run.
paper  5 repetitions of the settings the paper states numbers for; writes results/ntac_reproduction.json.
"""

import json
import sys
import time
import urllib.request
import zipfile

import numpy as np

from flyct import config
from flyct.ntac import seeded_ntac

DATA_URL = "https://github.com/BenJourdan/ntac/releases/download/v0.1.0/dynamic_data.zip"
DATA_MD5 = "a7cbde51764d18922edeabfaf3bb2c0e"
ROOT = config.DATA / "external" / "ntac"
BASE = ROOT / "flywire_data" / "dynamic_data" / "female_brain"

# (subset, side, seed fraction, what the paper says)
PAPER = [
    ("ol_intrinsic", "left", 0.02, "over 95% on the visual system with 2% of neurons labelled"),
    ("entire_visual_system", "left", 0.02, "over 95% on the visual system with 2% of neurons labelled"),
    ("entire_visual_system", "left", 0.05, None),
    ("central_brain", "left_and_right", 0.80, "87% on the central brain with 80% labelled"),
]


def fetch():
    from flyct.download import md5sum

    ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = ROOT / "dynamic_data.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(DATA_URL, zip_path)
    assert md5sum(zip_path) == DATA_MD5, "dynamic_data.zip checksum mismatch"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ROOT / "flywire_data")


def load(area, side):
    from ntac import FAFBData

    return FAFBData(edges_file=str(BASE / area / f"{side}_edges.csv"),
                    types_file=str(BASE / area / f"{side}_clusters.csv"))


def ours(data, train, k=5):
    gt = data.labels
    classes = np.unique(gt[train])
    rows, lab, _ = seeded_ntac(data.adj_csr, train, np.searchsorted(classes, gt[train]), len(classes), iters=15, lr=1.0, topk=k)
    top = np.full((data.n, k), None, dtype=object)
    top[train, 0] = gt[train]
    top[rows] = np.where(lab >= 0, classes[np.maximum(lab, 0)], None)
    return top


def split(data, frac, seed):
    train, test = data.test_train_split(train_size=frac, sampling_type="at_least_one_per_class", random_seed=seed)
    train, test = np.asarray(train), np.asarray(test)
    return train, test[data.labels[test] != data.unlabeled_symbol]


def agree(area="ol_columnar", side="left", frac=0.05, seed=5):
    from ntac import Ntac

    data = load(area, side)
    train, test = split(data, frac, seed)
    gt = data.labels
    t0 = time.perf_counter()
    mine = ours(data, train)[:, 0]
    t_mine = time.perf_counter() - t0
    masked = gt.copy()
    masked[np.setdiff1d(np.arange(data.n), train)] = data.unlabeled_symbol
    t0 = time.perf_counter()
    nt = Ntac(data=data, labels=masked, lr=1, topk=1)
    for _ in range(15):
        nt.step()
    ref = nt.get_partition()
    t_ref = time.perf_counter() - t0
    print(f"{area}/{side}, {frac:.0%} seeds, seed {seed}: flyct acc {(mine[test] == gt[test]).mean():.4f} ({t_mine:.0f}s), "
          f"reference acc {(ref[test] == gt[test]).mean():.4f} ({t_ref:.0f}s), "
          f"same prediction on {(mine[test] == ref[test]).mean():.2%} of test neurons")


def paper():
    out = []
    for area, side, frac, claim in PAPER:
        data = load(area, side)
        gt = data.labels
        acc, top5 = [], []
        for seed in range(5):
            train, test = split(data, frac, seed)
            top = ours(data, train)
            acc.append((top[test, 0] == gt[test]).mean())
            top5.append((top[test] == gt[test][:, None]).any(axis=1).mean())
        r = {"subset": area, "side": side, "seed_fraction": frac, "neurons": int(data.n),
             "acc_mean": round(float(np.mean(acc)), 4), "acc_sd": round(float(np.std(acc)), 4),
             "top5_mean": round(float(np.mean(top5)), 4), "paper_claim": claim}
        out.append(r)
        print(json.dumps(r))
    (config.ROOT / "results" / "ntac_reproduction.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    {"fetch": fetch, "agree": agree, "paper": paper}[sys.argv[1]]()
