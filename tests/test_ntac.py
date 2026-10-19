"""flyct.ntac against a brute-force weighted-Jaccard implementation on small random graphs."""

import numpy as np
import scipy.sparse as sp

from flyct.ntac import embed, nearest_seeds, seeded_ntac


def brute_force(A, seeds, seed_lab, k, iters):
    n = A.shape[0]
    part = np.zeros(n, np.int64)
    part[seeds] = seed_lab
    rows = np.setdiff1d(np.arange(n), seeds)
    for _ in range(iters):
        E = embed(A, A.T.tocsr(), part, k).toarray().astype(np.float64)
        F = E[seeds]
        for i in rows:
            sim = np.minimum(E[i], F).sum(1) / np.maximum(np.maximum(E[i], F).sum(1), 1e-12)
            part[i] = seed_lab[np.argmax(sim)] if sim.max() > 0 else np.bincount(seed_lab, minlength=k).argmax()
    return part[rows]


def random_graph(n=300, k=6, density=0.03, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, format="csr", random_state=rng,
                  data_rvs=lambda m: rng.integers(5, 40, m)).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()
    seeds = np.sort(rng.choice(n, n // 5, replace=False))
    return A, seeds, rng.integers(0, k, len(seeds)), k


def test_matches_brute_force():
    for s in range(3):
        A, seeds, lab, k = random_graph(seed=s)
        _, ours, _ = seeded_ntac(A, seeds, lab, k, iters=4)
        assert (ours[:, 0] == brute_force(A, seeds, lab, k, iters=4)).all()


def test_topk_is_unique_and_ordered():
    A, seeds, lab, k = random_graph(seed=7)
    part = np.zeros(A.shape[0], np.int64)
    part[seeds] = lab
    E = embed(A, A.T.tocsr(), part, k)
    rows = np.setdiff1d(np.arange(A.shape[0]), seeds)
    top, sim = nearest_seeds(E, rows, seeds, lab.astype(np.int64), k, 3, 0)
    for t, s in zip(top, sim):
        got = t[t >= 0]
        assert len(set(got)) == len(got)
        assert (np.diff(s[: len(got)]) <= 1e-7).all()
