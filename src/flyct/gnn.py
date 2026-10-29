"""GraphSAGE on the connectome, in plain PyTorch with sparse matmuls (no PyTorch Geometric).

Each layer mixes a neuron's own state with the weighted mean of its upstream partners' states and,
separately, of its downstream partners' states; the edge weight is log(1 + synapses). Two layers.

Seed labels are an input as well as the target (the "label trick"): every neuron gets an embedding of
its seed label, or of an extra "unknown" index when it has none. In each training step a random half of
the seeds show their label and the other half hide it and supply the loss, so the model can't learn to
copy a neuron's own label. At prediction time every seed shows its label and no scored neuron does. The
first layer's mean of partner label embeddings is a compressed partner-type profile, the signal NTAC
uses; the node features add what the neuron itself looks like.

Memory stays flat: the adjacency is two sparse matrices, and logits are computed only for the neurons
that need them (loss or prediction), in chunks, because an n x 8,840 cell-type logit matrix would not fit.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEGREE = ("out_degree", "out_syn", "in_degree", "in_syn")
LOGIT_BUDGET = 40_000_000  # floats of logits per training step or prediction chunk (~160 MB)


def feature_group(name: str) -> str:
    """degree (connectivity counts), nt (neurotransmitter shares) or region (neuropil synapse counts)."""
    if name in DEGREE:
        return "degree"
    return "nt" if "_nt_" in name else "region"


def node_features(data, use) -> torch.Tensor:
    """The chosen feature groups, log-scaled and standardised over all neurons (no labels involved).
    Neurotransmitter shares are NaN for neurons without outputs (or inputs); those become the mean, 0."""
    cols = [i for i, c in enumerate(data.feature_names) if feature_group(c) in use]
    X = np.log1p(data.X[:, cols])
    X = np.nan_to_num((X - np.nanmean(X, 0)) / np.maximum(np.nanstd(X, 0), 1e-6))
    X = np.clip(X, -10, 10)  # rare neuropils have tiny spread
    return torch.from_numpy(X.astype(np.float32)).to(DEVICE)


def _csr(W: sp.csr_matrix) -> torch.Tensor:
    W = W.tocsr()
    t = lambda a: torch.from_numpy(a.astype(np.int64))  # noqa: E731
    return torch.sparse_csr_tensor(t(W.indptr), t(W.indices), torch.from_numpy(W.data.astype(np.float32)),
                                   W.shape).to(DEVICE)


class MeanOp(torch.autograd.Function):
    """P @ h with the gradient P.T @ g from a stored CSR transpose: PyTorch's own sparse backward would
    transpose P on every step, which is several times slower than the product itself."""

    @staticmethod
    def forward(ctx, h, P, PT):
        ctx.PT = PT
        return P @ h

    @staticmethod
    def backward(ctx, g):
        return ctx.PT @ g, None, None


def mean_operator(A: sp.csr_matrix) -> tuple:
    """Row-normalised log(1 + synapses) weights: (P @ h)[i] = weighted mean of h over row i's partners.
    Returned as (P, P.T), both CSR tensors."""
    W = A.astype(np.float32, copy=True)
    W.data = np.log1p(W.data)
    deg = np.asarray(W.sum(axis=1)).ravel()
    W = sp.diags(np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)) @ W
    return _csr(W), _csr(W.T)


class SAGE(torch.nn.Module):
    def __init__(self, n_feat, n_cls, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.label = torch.nn.Embedding(n_cls + 1, hidden)  # index n_cls = unknown
        self.feat = torch.nn.Linear(n_feat, hidden) if n_feat else None
        self.convs = torch.nn.ModuleList(
            torch.nn.ModuleList([torch.nn.Linear(hidden, hidden), torch.nn.Linear(hidden, hidden, bias=False),
                                 torch.nn.Linear(hidden, hidden, bias=False)]) for _ in range(layers))
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(hidden) for _ in range(layers))
        self.out = torch.nn.Linear(hidden, n_cls)
        self.dropout = dropout

    def forward(self, X, lab, P_in, P_out):
        """Hidden state of every neuron; pass the rows that need them to logits()."""
        h = self.label(lab)
        if self.feat is not None:
            h = h + self.feat(X)
        for (w_self, w_in, w_out), norm in zip(self.convs, self.norms):
            h = F.relu(norm(w_self(h) + w_in(MeanOp.apply(h, *P_in)) + w_out(MeanOp.apply(h, *P_out))))
        return h

    def logits(self, h):
        # Dropout only here, on the few rows that get logits: dropout over every neuron's state costs
        # more per step on a CPU than a sparse product.
        return self.out(F.dropout(h, self.dropout, self.training))


def _topk(model, h, rows, k):
    chunk = max(256, LOGIT_BUDGET // model.out.out_features)
    conf, top = [], []
    for s in range(0, len(rows), chunk):
        c, t = torch.softmax(model.logits(h[rows[s:s + chunk]]), dim=1).topk(k, dim=1)
        conf.append(c)
        top.append(t)
    return torch.cat(conf), torch.cat(top)


def sage(data, k=5, use=("degree", "region", "nt"), hidden=128, epochs=400, lr=1e-2, eval_every=5,
         patience=8, seed=0, verbose=False):
    """Fit on seeds, early-stop on validation accuracy, predict data.scored."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t = lambda a: torch.from_numpy(a).to(DEVICE)  # noqa: E731
    classes, seed_lab = np.unique(data.y[data.seed], return_inverse=True)
    n_cls = len(classes)
    val = data.val[np.isin(data.y[data.val], classes)]
    val, y_val = t(val), t(np.searchsorted(classes, data.y[val]))
    X = node_features(data, use)
    P_out, P_in = mean_operator(data.A), mean_operator(data.A.T.tocsr())
    seeds, y_seed = t(data.seed), t(seed_lab)
    unknown = torch.full((len(data.y),), n_cls, dtype=torch.long, device=DEVICE)
    lab_all = unknown.clone()
    lab_all[seeds] = y_seed
    max_targets = max(256, LOGIT_BUDGET // n_cls)

    model = SAGE(X.shape[1], n_cls, hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        show = rng.random(len(data.seed)) < 0.5
        targets = np.flatnonzero(~show)
        if len(targets) > max_targets:
            targets = rng.choice(targets, max_targets, replace=False)
        lab = unknown.clone()
        lab[seeds[t(show)]] = y_seed[t(show)]
        targets = t(targets)
        loss = F.cross_entropy(model.logits(model(X, lab, P_in, P_out)[seeds[targets]]), y_seed[targets])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % eval_every:
            continue
        model.eval()
        with torch.no_grad():
            acc = (_topk(model, model(X, lab_all, P_in, P_out), val, 1)[1][:, 0] == y_val).float().mean().item()
        if verbose:
            print(f"  epoch {epoch:3d}  loss {loss.item():.3f}  val acc {acc:.4f}", flush=True)
        if acc > best:
            best, bad = acc, 0
            best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        conf, top = _topk(model, model(X, lab_all, P_in, P_out), t(data.scored), k)
    return classes[top.cpu().numpy()], conf[:, 0].cpu().numpy()
