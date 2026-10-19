"""Pinned data sources and project paths. Change a version here, nowhere else."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Override with FLYCT_DATA, e.g. on Kaggle where the repo and the data live in different places.
DATA = Path(os.environ.get("FLYCT_DATA", ROOT / "data"))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

# FlyWire connectome release 783, archived on Zenodo (CC-BY 4.0).
ZENODO_RECORD = "10676866"
ZENODO_FILES = {
    "proofread_connections_783.feather": "f48f972d262323a102aed49af1396b8a",
    "per_neuron_neuropil_count_pre_783.feather": "90fcdb42c1ba05ed92820840fa1e6ba0",
    "per_neuron_neuropil_count_post_783.feather": "bb5999f10920ade803d9f37097a43a56",
    "proofread_root_ids_783.npy": "e0e6c19732fd8c7a4e39a2d170105421",
}

# Labels: flywire_annotations, pinned to one tag. Codex mixes sources, so never use it for labels.
ANNOTATIONS_TAG = "v3.1.0"
ANNOTATIONS_COMMIT = "8587524c1748ce5ef2080822a2fc890fc03bf597"
ANNOTATIONS_FILE = "Supplemental_file1_neuron_annotations.tsv"
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    f"{ANNOTATIONS_COMMIT}/supplemental_files/{ANNOTATIONS_FILE}"
)

# Keep a neuron pair only if it has at least this many synapses summed across neuropils (Codex default).
SYNAPSE_THRESHOLD = 5

# Published reference numbers for sanity checks.
EXPECTED_NEURONS = 139_255
EXPECTED_EDGES_APPROX = 2_700_000


def zenodo_url(name: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
