# ADR-002: Graph neural network design, and what the ablations showed

Status: accepted, 2026-10-30

## Context

Week 3 asks whether a graph neural network beats NTAC, by how much, and where. Two constraints shaped the
design. First, the Week 1 measurement: only 1.1% of edges join two neurons of the same type, but same-type
neurons have very similar partner-type profiles (median cosine 0.86, against 0.00 for different types).
A GNN that only smooths features over neighbours would be averaging over neurons of *other* types.
Second, the hardware: a 4-core, 5.9 GB laptop, with Kaggle's T4 GPU for full grids.

## Decision

**Model** (`src/flyct/gnn.py`): two-layer GraphSAGE written in plain PyTorch, without PyTorch Geometric.

- Each layer combines a neuron's own state with two separate weighted means, one over upstream partners
  and one over downstream partners: `h' = ReLU(LayerNorm(W_self h + W_in P_in h + W_out P_out h))`.
  Edge weight is log(1 + synapses), row-normalised, so `P_in` and `P_out` are sparse mean operators.
- **Seed labels as input (the label trick).** Every neuron starts from an embedding of its seed label, or
  of an "unknown" index if it isn't a seed. In each training step a random half of the seeds show their
  label and the other half hide it and supply the loss, so the model can't learn to copy a neuron's own
  label. At prediction time every seed shows its label and no scored neuron does (ADR-001's rule:
  partner-type information comes from seeds only). The first layer's mean of partner label embeddings is
  a learned, compressed version of NTAC's partner-type profile.
- **Node features** are added to the input embedding: log-scaled, standardised over all neurons (no
  labels involved), clipped to ±10. The neurotransmitter shares are NaN for the 1,250 neurons with no
  outputs (2,165 with no inputs) and are filled with the mean.
- Fixed configuration, not tuned: hidden size 128, dropout 0.3 on the logit rows, Adam at 1e-2, up to 400
  full-batch steps, validation accuracy every 5 steps, early stopping after 8 checks without improvement.

**Memory and speed.** The adjacency is held as two CSR tensors plus their transposes; a custom autograd
function uses the stored transpose for the backward pass (PyTorch's COO sparse product transposes on every
step and was about 6× slower). Logits are computed only for the rows that need them, in chunks, so the
8,840-way task never materialises a neurons × types matrix. Peak memory is 1.3 GB on the laptop. A step
takes 3–5 s on the laptop CPU (a run takes 10–15 min), against 25–45 s for a whole run on a Kaggle T4, so
the 198-run grid ran on Kaggle in 1.8 GPU-hours.

**Ablations**, all on the same splits as every other model:

| Method | Inputs |
| --- | --- |
| `sage_conn` | seed labels + degree counts (in/out partners and synapses) |
| `sage_region` | + synapses per neuropil, mirrored to ipsi/contra (brain region) |
| `sage` | + neurotransmitter mix of outputs and inputs |

## Results

Test accuracy (%), mean over 5 splits; hemisphere transfer is one split each way.

| Task | Setting | Best non-GNN | `sage_conn` | `sage_region` | `sage` |
| --- | --- | --- | --- | --- | --- |
| Super class | 20% seeds | 97.1 (LightGBM) | 97.0 | 98.2 | **98.4** |
| Super class | L→R | 96.8 (LightGBM) | 72.6 | 98.3 | **98.7** |
| Class | 1% seeds | 86.0 (LightGBM) | 84.4 | 93.2 | **93.4** |
| Class | 20% seeds | 90.7 (LightGBM) | 95.4 | 98.1 | **98.2** |
| Class | L→R | 91.4 (LightGBM) | 47.8 | 98.3 | **98.4** |
| Cell type | 1% seeds | **58.9** (NTAC) | 22.7 | 40.2 | 42.4 |
| Cell type | 20% seeds | **92.1** (NTAC) | 81.9 | 87.3 | 87.9 |
| Cell type | 80% seeds | **94.2** (NTAC) | 90.0 | 90.7 | 91.5 |
| Cell type | L→R | 76.3 (NTAC) | 28.6 | 82.8 | **86.2** |
| Cell type | R→L | **86.3** (NTAC) | 25.5 | 77.2 | 81.1 |

Cell type at 20% seeds, by region: optic lobe 96.2 (NTAC 96.3), central brain 66.7 (NTAC 84.7),
visual projection 91.3 (NTAC 95.7); types with 2 neurons 67.6 (NTAC 92.4). Macro F1 is 55.1 against
NTAC's 82.3, and it barely moves with more seeds (50.6 at 1%, 54.7 at 80%).

## What the ablations showed

1. **Labels as input plus message passing is the strongest model for coarse labels.** `sage` beats
   LightGBM on super class and class in every setting; on class it gains 7.5 points at 20% seeds and
   7.4 at 1%. Neither input alone gets there: LightGBM has the features but not the partners' labels, and
   the partner-profile models have the labels but are linear and break under hemisphere transfer.
2. **Brain-region features are what make hemisphere transfer work.** Without them, class accuracy from
   left to right falls to 47.8% (`sage_conn`); with them it is 98.3%. The cell-type optic lobe shows the
   same: 30.8% → 90.8% from left to right. The opposite hemisphere has few labelled partners (only
   cross-midline ones), so a neuron's own synapse locations, mirrored to ipsi/contra, carry the transfer.
3. **Neurotransmitter features add little:** 0.1–1.1 points of accuracy on super class and class, and
   0.6–3.9 on cell type, largest under hemisphere transfer.
4. **On cell type the GNN loses to NTAC, and the loss sits where types are rare or defined by partners.**
   The gap is 18 points in the central brain and 25 points on 2-neuron types, but near zero on the optic
   lobe and on types over 100 neurons. The likely cause is the bottleneck the plan predicted: NTAC keeps
   one dimension per type and matches each neuron to its nearest seed, while the GNN squeezes 8,840 types
   into a 128-number embedding, averages it over partners, and must learn a softmax over 8,840 classes,
   most of them with 1–3 seeds. That macro F1 hardly moves between one seed per type (6.4% of neurons) and 80% seeds points the same way.
5. **The one cell-type setting the GNN wins is left → right transfer (86.2 vs 76.3),** and it wins it on
   the optic lobe (94.0 vs 69.6, where NTAC is weak in that direction), while still losing the central
   brain (69.9 vs 89.5).

## Consequences

- **Model choice per task:** GraphSAGE with all features for super class and class; NTAC stays the
  cell-type model. The README reports both.
- The GNN is not tuned. Tuning would most likely help cell type (wider embedding, more steps), but the
  central-brain and rare-type gaps are large enough that a conclusion is safe at this configuration.
- Follow-ups, in order of what they would teach:
  1. The plan's open question: add brain-region and neurotransmitter features to the partner-profile
     model, and see whether they help where it is weakest (central brain).
  2. Test the bottleneck explanation directly: feed NTAC's full partner-type profile into the GNN as an
     input, or widen the label embedding, and check whether the central-brain gap closes.

## Alternatives considered

- **PyTorch Geometric.** Its `SAGEConv` would do the same aggregation, but it adds a dependency that is
  awkward to install on Windows and Kaggle images alike, and our layer is a dozen lines. The custom
  backward pass (stored transpose) was the one non-trivial piece, and PyG would not have provided it.
- **Mini-batch training with neighbour sampling** (`NeighborLoader`). Not needed: full-batch training fits
  in 1.3 GB, because logits are only computed for the rows that need them.
- **One-hot seed labels as features.** An n × 8,841 matrix, even sparse, feeds a dense first layer of
  8,841 × hidden weights. An `nn.Embedding` is that same layer, without building the matrix.
- **Training on all seeds' labels at once, with no masking.** Each seed would see its own label and the
  model would learn to copy its own label, an input no test neuron has.
- **GAT (attention).** Left as a stretch goal: the cell-type gap looks like a capacity and rare-class
  problem, which attention over neighbours doesn't address.
