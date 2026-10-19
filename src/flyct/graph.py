"""The thresholded connectome as a sparse matrix in the row order of labels.parquet."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyct import config


def adjacency(root_ids: np.ndarray) -> sp.csr_matrix:
    """A[i, j] = synapses from neuron i to neuron j (pairs with >= SYNAPSE_THRESHOLD synapses)."""
    edges = pd.read_parquet(config.PROCESSED / "edges.parquet", columns=["pre_root_id", "post_root_id", "syn_count"])
    pos = pd.Series(np.arange(len(root_ids)), index=root_ids)
    src = pos.loc[edges.pre_root_id].to_numpy()
    dst = pos.loc[edges.post_root_id].to_numpy()
    n = len(root_ids)
    return sp.csr_matrix((edges.syn_count.to_numpy(np.float32), (src, dst)), shape=(n, n))
