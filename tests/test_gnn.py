"""flyct.gnn on a small synthetic graph where a neuron's type is set by its partners' types, not its own
features, and same-type neurons never connect: only the label input can solve it."""

import numpy as np
import scipy.sparse as sp

from flyct import splits
from flyct.gnn import mean_operator, sage
from flyct.run import TaskData


def partner_graph(n=1200, k=6, out_deg=8, seed=0):
    """Type t sends synapses only to types (t + 1) % k and (t + 2) % k."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    by_type = [np.flatnonzero(y == c) for c in range(k)]
    src, dst = [], []
    for i in range(n):
        for c in ((y[i] + 1) % k, (y[i] + 2) % k):
            src += [i] * out_deg
            dst += list(rng.choice(by_type[c], out_deg))
    A = sp.csr_matrix((rng.integers(5, 40, len(src)).astype(np.float32), (src, dst)), shape=(n, n))
    roles = rng.choice([splits.SEED, splits.VAL, splits.TEST], n, p=[0.3, 0.1, 0.6]).astype(np.int8)
    X = rng.random((n, 4)).astype(np.float32)  # pure noise
    return TaskData("t", np.arange(n), y.astype(object), roles, X, ["out_degree", "out_syn", "in_degree", "in_syn"], A)


def test_mean_operator_rows_are_weighted_means():
    A = sp.csr_matrix(np.array([[0, 3, 1], [0, 0, 0], [7, 0, 0]], np.float32))
    P, PT = mean_operator(A)
    W = np.log1p(A.toarray())
    W[0] /= W[0].sum()
    W[2] /= W[2].sum()
    np.testing.assert_allclose(P.to_dense().cpu().numpy(), W, rtol=1e-6)
    np.testing.assert_allclose(PT.to_dense().cpu().numpy(), W.T, rtol=1e-6)


def test_sage_learns_types_from_partner_labels():
    data = partner_graph()
    topk, conf = sage(data, k=3, use=("degree",), hidden=32, epochs=150)
    assert topk.shape == (len(data.scored), 3) and conf.shape == (len(data.scored),)
    acc = (topk[len(data.val):, 0] == data.y[data.test]).mean()
    assert acc > 0.9, acc  # chance is 1/6
