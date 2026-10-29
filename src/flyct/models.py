"""Baseline models. Each takes the labels of seed neurons only and returns, for every neuron to score,
its top-k predicted labels (best first) and the probability of the top one (None if the model has no
probabilities).

The signature is the same for all models: `fit_predict(data, k) -> (topk, conf)`, where `data` is a
`flyct.run.TaskData` and the returned arrays cover `data.scored` (validation + test neurons) in order.
"""

import lightgbm as lgb
import numpy as np


def majority(data, k=5):
    """Floor: always predict the most common seed classes."""
    classes, counts = np.unique(data.y[data.seed], return_counts=True)
    order = np.argsort(-counts, kind="stable")[:k]
    n = len(data.scored)
    topk = np.tile(classes[order], (n, 1))
    conf = np.full(n, counts[order[0]] / counts.sum())
    return topk, conf


def lightgbm(data, k=5):
    """Gradient-boosted trees on per-neuron features. Super class and class only: a multiclass model
    grows one tree per class per round, so 8,840 cell types are impractical."""
    classes, y_seed = np.unique(data.y[data.seed], return_inverse=True)
    X = data.X
    # Validation neurons whose class has no seed (possible in the hemisphere split) can't be scored
    # by any model, so they are left out of early stopping.
    val = data.val[np.isin(data.y[data.val], classes)]
    y_val = np.searchsorted(classes, data.y[val])
    params = {
        "objective": "multiclass", "num_class": len(classes), "learning_rate": 0.1, "num_leaves": 31,
        "min_data_in_leaf": 10, "feature_fraction": 0.8, "num_threads": 4, "verbose": -1, "seed": 0,
    }
    train = lgb.Dataset(X[data.seed], y_seed, free_raw_data=True)
    valid = lgb.Dataset(X[val], y_val, reference=train)
    booster = lgb.train(params, train, num_boost_round=500, valid_sets=[valid],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
    proba = booster.predict(X[data.scored], num_iteration=booster.best_iteration)
    top = np.argsort(-proba, axis=1)[:, :k]
    return classes[top], proba.max(axis=1)


def ntac(data, k=5):
    """Seeded NTAC on the whole-brain graph (connectivity only, no features). Its score is a
    similarity, not a probability, so it reports no confidence."""
    from flyct.ntac import seeded_ntac

    classes, seed_lab = np.unique(data.y[data.seed], return_inverse=True)
    rows, lab, _ = seeded_ntac(data.A, data.seed, seed_lab, len(classes), iters=15, lr=1.0, topk=k)
    lab = lab[np.searchsorted(rows, data.scored)]
    topk = np.where(lab >= 0, classes[np.maximum(lab, 0)], None)
    return topk, None


def partner_profile(A, part, k):
    """Synapses received from and sent to each class, log-scaled and L2-normalised per neuron.
    `part` holds a class per neuron, or -1 where unknown (those partners are left out)."""
    from sklearn.preprocessing import normalize

    from flyct.ntac import embed

    E = embed(A, A.T.tocsr(), part, k)
    E.data = np.log1p(E.data)
    return normalize(E, norm="l2")


def profile_lr(data, k=5, rounds=1):
    """Logistic regression on the partner-type profile, trained on seeds. With rounds > 1, non-seed
    partners are filled in with the previous round's predictions and the model is refitted, like
    NTAC's iterations. Super class and class only on a laptop (dense k x 2k weights)."""
    from sklearn.linear_model import LogisticRegression

    classes, seed_lab = np.unique(data.y[data.seed], return_inverse=True)
    n_cls = len(classes)
    part = np.full(len(data.y), -1, np.int64)
    part[data.seed] = seed_lab
    val = data.val[np.isin(data.y[data.val], classes)]
    y_val = np.searchsorted(classes, data.y[val])
    rest = np.setdiff1d(np.arange(len(data.y)), data.seed)
    for _ in range(rounds):
        X = partner_profile(data.A, part, n_cls)
        best = None
        for C in (0.1, 1.0, 10.0):  # chosen on validation accuracy
            clf = LogisticRegression(C=C, max_iter=500, tol=1e-3).fit(X[data.seed], seed_lab)
            acc = (clf.predict(X[val]) == y_val).mean()
            if best is None or acc > best[0]:
                best = (acc, clf)
        clf = best[1]
        part[rest] = clf.predict(X[rest])
    proba = clf.predict_proba(X[data.scored])
    top = np.argsort(-proba, axis=1)[:, :k]
    return classes[clf.classes_[top]], proba.max(axis=1)


def profile_lr_iter(data, k=5):
    return profile_lr(data, k, rounds=3)


def profile_linear(data, k=5):
    """profile_lr in PyTorch, for cell type (see torch_models)."""
    from flyct.torch_models import profile_linear

    return profile_linear(data, k)


def profile_linear_iter(data, k=5):
    from flyct.torch_models import profile_linear_iter

    return profile_linear_iter(data, k)


def sage(data, k=5):
    """GraphSAGE with seed labels as input and every node feature (see gnn)."""
    from flyct.gnn import sage

    return sage(data, k)


def sage_conn(data, k=5):
    """Ablation: connectivity only (degree counts, no neuropil or neurotransmitter features)."""
    from flyct.gnn import sage

    return sage(data, k, use=("degree",))


def sage_region(data, k=5):
    """Ablation: connectivity plus neuropil (brain-region) features, no neurotransmitter features."""
    from flyct.gnn import sage

    return sage(data, k, use=("degree", "region"))


MODELS = {"majority": majority, "lightgbm": lightgbm, "ntac": ntac,
          "profile_lr": profile_lr, "profile_lr_iter": profile_lr_iter,
          "profile_linear": profile_linear, "profile_linear_iter": profile_linear_iter,
          "sage_conn": sage_conn, "sage_region": sage_region, "sage": sage}
GRAPH_MODELS = {"ntac", "profile_lr", "profile_lr_iter", "profile_linear", "profile_linear_iter",
                "sage_conn", "sage_region", "sage"}
