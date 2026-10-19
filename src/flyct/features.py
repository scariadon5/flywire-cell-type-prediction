"""Feature matrix for the feature-based models, with neuropil columns mirrored to a side-free form.

Neuropil columns come in left/right pairs (`pre_lo_l`, `pre_lo_r`), so the same cell type on the two
hemispheres looks like two different feature patterns. Mirroring renames each pair to `_ipsi` / `_contra`,
where ipsi is the neuron's own dominant hemisphere: the side holding more of its synapses. The dominant
side is computed from the features themselves, never from the annotation `side` column, so no label
information enters the feature matrix.
"""

import numpy as np
import pandas as pd

from flyct import config


def mirror_neuropils(feats: pd.DataFrame) -> pd.DataFrame:
    cols = set(feats.columns)
    pairs = sorted(c[:-2] for c in cols if c.endswith("_l") and c[:-2] + "_r" in cols)
    left = feats[[p + "_l" for p in pairs]].to_numpy(dtype=np.float32)
    right = feats[[p + "_r" for p in pairs]].to_numpy(dtype=np.float32)
    flip = (right.sum(axis=1) > left.sum(axis=1))[:, None]
    ipsi = np.where(flip, right, left)
    contra = np.where(flip, left, right)
    out = feats.drop(columns=[p + s for p in pairs for s in ("_l", "_r")])
    mirrored = pd.DataFrame(
        np.hstack([ipsi, contra]),
        columns=[p + "_ipsi" for p in pairs] + [p + "_contra" for p in pairs],
        index=feats.index,
    )
    lateral = left.sum(axis=1) + right.sum(axis=1)
    mirrored["contra_share"] = np.divide(contra.sum(axis=1), lateral, out=np.zeros_like(lateral), where=lateral > 0)
    return pd.concat([out, mirrored], axis=1)


def load(mirror: bool = True) -> pd.DataFrame:
    """Features indexed by root_id, float32, in the same row order as labels.parquet."""
    feats = pd.read_parquet(config.PROCESSED / "features.parquet").set_index("root_id")
    if mirror:
        feats = mirror_neuropils(feats)
    return feats.astype(np.float32)
