"""Seeded NTAC ("Connectivity is all you need", Jourdan et al.), reimplemented to fit in a few GB of RAM.

Each neuron is described by a 2k-dimensional vector: synapses received from each of the k classes,
then synapses sent to each class, where a neuron's class is its seed label or its current prediction.
Every non-seed neuron takes the class of its most similar seed under weighted Jaccard similarity,
sum_d min(a_d, b_d) / sum_d max(a_d, b_d); the vectors are recomputed from the new predictions, and
this repeats for a fixed number of iterations. Seeds never change.

Their package builds a dense (neurons x seeds) similarity matrix, about 15 GB for the whole brain at
20% seeds. Here a numba kernel walks, for each neuron, only the seeds that share a nonzero dimension
with it, and keeps the best similarity per class, so memory stays at a few MB per thread.

Matches the reference behaviour: unlabelled neurons start in class 0 (the first class in sorted
order), and with lr < 1 the vectors are blended as (1 - lr) * old + lr * new.
"""

import numpy as np
import scipy.sparse as sp
from numba import get_num_threads, njit, prange


def embed(A: sp.csr_matrix, AT: sp.csr_matrix, part: np.ndarray, k: int) -> sp.csr_matrix:
    """[synapses in from each class | synapses out to each class], one row per neuron."""
    n = len(part)
    assigned = np.flatnonzero(part >= 0)
    P = sp.csr_matrix((np.ones(len(assigned), np.float32), (assigned, part[assigned])), shape=(n, k))
    return sp.hstack([AT @ P, A @ P], format="csr", dtype=np.float32)


@njit(parallel=True, cache=True)
def _nearest(r_ptr, r_idx, r_val, r_sum, s_ptr, s_idx, s_val, s_sum, s_lab, n_lab, topk, fallback):
    n = len(r_ptr) - 1
    m = len(s_sum)
    out_lab = np.full((n, topk), -1, np.int64)
    out_sim = np.zeros((n, topk), np.float32)
    blocks = min(n, 8 * get_num_threads())
    for blk in prange(blocks):
        inter = np.zeros(m, np.float32)   # sum_d min(a_d, b_d) per touched seed
        touched = np.empty(m, np.int64)
        best = np.full(n_lab, -1.0, np.float32)  # best similarity per class; -1 = not seen
        best_j = np.zeros(n_lab, np.int64)       # seed achieving it, for tie-breaking
        seen = np.empty(n_lab, np.int64)
        for r in range(blk, n, blocks):
            nt = 0
            for p in range(r_ptr[r], r_ptr[r + 1]):
                d = r_idx[p]
                a = r_val[p]
                for q in range(s_ptr[d], s_ptr[d + 1]):
                    j = s_idx[q]
                    if inter[j] == 0.0:
                        touched[nt] = j
                        nt += 1
                    b = s_val[q]
                    inter[j] += a if a < b else b
            ns = 0
            for t in range(nt):
                j = touched[t]
                x = inter[j]
                inter[j] = 0.0
                s = x / (r_sum[r] + s_sum[j] - x)
                c = s_lab[j]
                if best[c] < 0:
                    seen[ns] = c
                    ns += 1
                    best[c] = s
                    best_j[c] = j
                elif s > best[c] or (s == best[c] and j < best_j[c]):
                    best[c] = s
                    best_j[c] = j
            # Exact ties are common (e.g. while unlabelled neurons all sit in class 0). Like the
            # reference argmax, the seed with the lowest index wins.
            for t in range(topk):
                bi = -1
                bs = -1.0
                bj = m
                for u in range(ns):
                    c = seen[u]
                    if best[c] > bs or (best[c] == bs and best_j[c] < bj):
                        bs = best[c]
                        bj = best_j[c]
                        bi = u
                if bi < 0:
                    break
                out_lab[r, t] = seen[bi]
                out_sim[r, t] = bs
                best[seen[bi]] = -2.0  # taken; below -1 so it is never picked again
            for u in range(ns):
                best[seen[u]] = -1.0
            if out_lab[r, 0] < 0:  # no partner shares a class with any seed: no evidence at all
                out_lab[r, 0] = fallback
    return out_lab, out_sim


def nearest_seeds(E: sp.csr_matrix, rows: np.ndarray, seeds: np.ndarray, seed_lab: np.ndarray,
                  n_lab: int, topk: int, fallback: int):
    R = E[rows]
    S = E[seeds].tocsc()  # per dimension: which seeds have it, and their value
    return _nearest(R.indptr, R.indices, R.data, np.asarray(R.sum(axis=1)).ravel().astype(np.float32),
                    S.indptr, S.indices, S.data, np.asarray(S.sum(axis=1)).ravel().astype(np.float32),
                    seed_lab, n_lab, topk, fallback)


def seeded_ntac(A: sp.csr_matrix, seeds: np.ndarray, seed_lab: np.ndarray, n_lab: int,
                iters: int = 15, lr: float = 1.0, topk: int = 5, verbose: bool = False):
    """Predict a class for every non-seed neuron.

    A         (n, n) weighted adjacency, A[i, j] = synapses i -> j
    seeds     indices of seed neurons;  seed_lab their classes in 0..n_lab-1
    Returns (rows, topk_labels, topk_similarity) for the non-seed rows.
    """
    n = A.shape[0]
    A = A.tocsr().astype(np.float32)
    AT = A.T.tocsr()
    order = np.argsort(seeds)
    seeds = np.asarray(seeds)[order]
    seed_lab = np.asarray(seed_lab, np.int64)[order]
    rows = np.setdiff1d(np.arange(n), seeds)
    part = np.zeros(n, np.int64)
    part[seeds] = seed_lab
    fallback = int(np.bincount(seed_lab, minlength=n_lab).argmax())
    E = embed(A, AT, part, n_lab)
    for it in range(iters):
        lab, sim = nearest_seeds(E, rows, seeds, seed_lab, n_lab, topk, fallback)
        changed = int((part[rows] != lab[:, 0]).sum())
        part[rows] = lab[:, 0]
        new = embed(A, AT, part, n_lab)
        E = new if lr == 1.0 else ((1 - lr) * E + lr * new).tocsr()
        if verbose:
            print(f"  iter {it + 1}: {changed:,} changed")
        if changed == 0 and lr == 1.0:
            break
    return rows, lab, sim
