"""Partner-profile logistic regression in PyTorch, for the 8,840-way cell-type task.

Same model as models.profile_lr (a linear softmax over the partner-type profile), but trained with
mini-batches on sparse inputs, so it scales to thousands of classes. The linear layer is an
EmbeddingBag in sum mode weighted by the profile values, which computes x @ W exactly but gives a
sparse gradient: each step updates only the rows of W for profile dimensions present in the batch,
instead of all 2k x k weights. W alone is about 625 MB for cell type, and SparseAdam keeps two more
copies, so cell-type runs go to Kaggle. Runs on a GPU when one is available.
"""

import numpy as np
import scipy.sparse as sp
import torch

from flyct.models import partner_profile

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SparseLinear(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.W = torch.nn.EmbeddingBag(n_in, n_out, mode="sum", sparse=True)
        torch.nn.init.zeros_(self.W.weight)
        self.bias = torch.nn.Parameter(torch.zeros(n_out))

    def forward(self, X: sp.csr_matrix, rows: np.ndarray):
        b = X[rows]
        t = lambda a, dt: torch.from_numpy(a.astype(dt)).to(DEVICE)  # noqa: E731
        return self.W(t(b.indices, np.int64), t(b.indptr[:-1], np.int64),
                      per_sample_weights=t(b.data, np.float32)) + self.bias


def _predict(model, X, rows, chunk=8192):
    with torch.no_grad():
        out = [model(X, rows[s:s + chunk]).cpu() for s in range(0, len(rows), chunk)]
    return torch.cat(out) if out else torch.empty(0, model.bias.shape[0])


def profile_linear(data, k=5, rounds=1, epochs=40, batch=2048, lr=1e-2, patience=4, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    classes, seed_lab = np.unique(data.y[data.seed], return_inverse=True)
    n_cls = len(classes)
    part = np.full(len(data.y), -1, np.int64)
    part[data.seed] = seed_lab
    val = data.val[np.isin(data.y[data.val], classes)]
    y_val = torch.from_numpy(np.searchsorted(classes, data.y[val]))
    y_seed = torch.from_numpy(seed_lab).to(DEVICE)
    rest = np.setdiff1d(np.arange(len(data.y)), data.seed)

    for _ in range(rounds):
        X = partner_profile(data.A, part, n_cls).astype(np.float32).tocsr()
        model = SparseLinear(X.shape[1], n_cls).to(DEVICE)
        opts = [torch.optim.SparseAdam(list(model.W.parameters()), lr=lr), torch.optim.Adam([model.bias], lr=lr)]
        best, best_state, bad = -1.0, None, 0
        for _ in range(epochs):
            order = rng.permutation(len(data.seed))
            for s in range(0, len(order), batch):
                b = order[s:s + batch]
                loss = torch.nn.functional.cross_entropy(model(X, data.seed[b]), y_seed[b])
                for o in opts:
                    o.zero_grad()
                loss.backward()
                for o in opts:
                    o.step()
            acc = (_predict(model, X, val).argmax(1) == y_val).float().mean().item()
            if acc > best:
                best, bad = acc, 0
                best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        model.load_state_dict(best_state)
        part[rest] = _predict(model, X, rest).argmax(1).numpy()

    proba = torch.softmax(_predict(model, X, data.scored), dim=1)
    conf, top = proba.topk(k, dim=1)
    return classes[top.numpy()], conf[:, 0].numpy()


def profile_linear_iter(data, k=5):
    return profile_linear(data, k, rounds=3)
