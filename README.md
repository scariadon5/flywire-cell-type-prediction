# FlyWire cell-type prediction

Semi-supervised node classification on the FlyWire whole-brain connectome of the fruit fly:
139,255 neurons, about 2.7 million weighted, directed connections, and thousands of cell types.
Given a fraction of labelled neurons, predict the type of every other neuron from its wiring.
Compare feature models, a graph neural network and the published NTAC method, and ask whether
non-connectivity features (where synapses sit, predicted neurotransmitters) help where accuracy is lowest.

> **Status: Week 1 done (data pipeline and exploration). Next: splits and baselines.** Results table and
> explorer come in later weeks; see [docs/project-plan.md](docs/project-plan.md).

## Quickstart

```bash
py -3.12 -m venv .venv          # Linux/Kaggle: python3 -m venv .venv
gmake setup                     # Windows: gmake from Strawberry Perl, in Git Bash. Linux: make
gmake data                      # download + verify -> Parquet -> clean graph -> sanity checks
gmake explore                   # execute notebooks/01_exploration.ipynb
```

`make data` is idempotent. Verified downloads are skipped, and an interrupted download resumes.
Zenodo throttles single connections heavily on some networks, so files are fetched as 16 parallel byte ranges.
Peak memory stays under about 3 GB: Feather is streamed to Parquet one record batch at a time,
and DuckDB (capped at 2 GB, spilling to disk) does all the aggregation.

## Data card

### Sources

| File | Source | Version | md5 |
| --- | --- | --- | --- |
| `proofread_connections_783.feather` | [Zenodo 10676866](https://zenodo.org/records/10676866) | release 783 | `f48f972d262323a102aed49af1396b8a` |
| `per_neuron_neuropil_count_pre_783.feather` | Zenodo 10676866 | release 783 | `90fcdb42c1ba05ed92820840fa1e6ba0` |
| `per_neuron_neuropil_count_post_783.feather` | Zenodo 10676866 | release 783 | `bb5999f10920ade803d9f37097a43a56` |
| `proofread_root_ids_783.npy` | Zenodo 10676866 | release 783 | `e0e6c19732fd8c7a4e39a2d170105421` |
| `Supplemental_file1_neuron_annotations.tsv` | [flywire_annotations](https://github.com/flyconnectome/flywire_annotations) | tag `v3.1.0` (commit `8587524`) | see `data/checksums.md5` |

All pins live in [`src/flyct/config.py`](src/flyct/config.py). The individual-synapse table
(`flywire_synapses_783.feather`, 9.5 GB) is deliberately not used.

**Licence:** connectivity data CC-BY 4.0. Annotations: see the flywire_annotations repository.

### Processing (`src/flyct/build.py`)

1. Keep only the 139,255 proofread neurons.
2. **Sum synapses across neuropils first, then threshold** at ≥ 5 synapses per directed pair (the Codex default).
   The raw file has one row per pair *per neuropil*, so thresholding first would wrongly drop pairs whose synapses are spread out.
3. Edge neurotransmitter probabilities are the synapse-weighted mean over that pair's neuropil rows.

### Outputs (`data/processed/`)

| File | Rows | Contents |
| --- | --- | --- |
| `edges.parquet` | one per directed pair | `pre_root_id`, `post_root_id`, `syn_count`, six NT probabilities |
| `features.parquet` | one per neuron | degree and synapse totals (thresholded graph); synapse-weighted NT mix of outgoing and incoming synapses; synapses per neuropil for outputs (`pre_*`) and inputs (`post_*`), counting all synapses |
| `labels.parquet` | one per neuron | `super_class`, `cell_class`, `cell_type` (targets) and `side` (for the hemisphere split only) |

**Features and labels are separate files by design.** Nothing that builds a feature matrix needs
to open `labels.parquet`, and `make check` fails if any annotation column appears in `features.parquet`.
Columns never used as features: `hemibrain_type`, `fbbt_id` / `vfb_id` (these map to type),
lineages, `top_nt`/`known_nt`, synonyms, soma/position coordinates, and everything else in the annotation file.

### Counts (checked by `make check` against [`data/build_report.json`](data/build_report.json))

| | Count | Published reference |
| --- | --- | --- |
| Proofread neurons | 139,255 | 139,255 (Dorkenwald et al. 2024) |
| Directed edges, ≥ 5 synapses | 2,700,513 | ~2.7 million |
| Synapses in kept edges | 34,153,566 | |
| Neurons with a super class / class / cell type | 139,241 / 107,511 / 137,714 | |
| Distinct cell types (annotations v3.1.0) | 8,840 | 8,453 in Schlegel et al. 2024; later releases added types |
| Feature columns | 176 (+ `root_id`) | |

Peak memory on a 5.9 GB laptop: 0.87 GB (convert), 2.05 GB (build).

### Week 1 findings ([notebook](notebooks/01_exploration.ipynb))

![Same-type neurons share partner types](docs/figures/partner_profile_similarity.png)

- **Same-type neurons share partner types but rarely connect to each other.** 1.1% of edges join same-type
  neurons, yet same-type pairs have a median partner-type-profile cosine of 0.86 (0.00 for different types).
  This favours NTAC-style features, and means a GNN needs seed labels as input to compete.
- **Extreme label tail.** 51% of types have exactly 2 neurons; the 150 types with more than 100 neurons hold 69% of all neurons.
- **Class labels are missing for 68% of central-brain neurons**, so class-level results mainly reflect optic and sensory neurons.
- **5,074 neurons have no partner at ≥ 5 synapses** (mostly sensory). Connectivity alone can't type them.

### Known caveats

- **Left/right was flipped in the original imaging.** Annotations were corrected; coordinates were not.
  Hemisphere logic uses the `side` column only. Neuropil names carry `_L`/`_R` suffixes, so hemisphere-transfer
  models need features mirrored to "same side / other side" (Week 2).
- **Some labels were defined from connectivity.** Optic-lobe types were largely typed by their wiring, so high
  accuracy there partly recovers the labelling procedure. All results are reported by region.
- **Annotations are pinned.** v3.0.0 → v3.1.0 → v3.2.0 each retyped neurons. v3.2.0 (Sept 2026) only retypes
  gustatory sensory neurons. We stay on v3.1.0 so numbers are stable; Codex mixes sources and is not used for labels.

## Repository layout

```
src/flyct/config.py       pinned versions, checksums, paths
src/flyct/download.py     parallel, resumable, checksum-verified download
src/flyct/convert.py      Feather -> Parquet, one record batch at a time
src/flyct/build.py        DuckDB: edges, features, labels
src/flyct/checks.py       sanity checks + reproducibility check
notebooks/                exploration (01_exploration.ipynb is generated by make_01_exploration.py)
docs/                     project plan, figures
```

## Citations

Connectivity and annotations come from the FlyWire Consortium. As the annotation repository asks, please cite:

- Dorkenwald, S. et al. *Neuronal wiring diagram of an adult brain.* Nature (2024). [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)
- Schlegel, P. et al. *Whole-brain annotation and multi-connectome cell typing of Drosophila.* Nature (2024). [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5)
- Matsliah, A. et al. *Neuronal parts list and wiring diagram for a visual system.* Nature (2024). [doi:10.1038/s41586-024-07981-1](https://doi.org/10.1038/s41586-024-07981-1)
- Berg, S. et al. *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.* bioRxiv (2025). [doi:10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999)

Baseline method: *Connectivity is all you need: inferring neuronal types with NTAC* ([code](https://github.com/BenJourdan/ntac)).
