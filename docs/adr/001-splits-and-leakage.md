# ADR-001: Split design and leakage rules

Status: accepted, 2026-10-19

## Context

The task is semi-supervised: some neurons are labelled ("seeds") and a model labels the rest. Whether a
number means anything depends on which neurons are seeds, what the model may look at, and that every
model sees the same split. We also want results that sit next to NTAC's published numbers.

## Decision

**Three tasks**, one per label level: `super_class` (10 classes), `cell_class` (49), `cell_type` (8,840).
A neuron without a label at a level stays in the graph for that task but is never a seed and never scored.

**Two settings**, saved once by `make splits` to `data/splits/<task>.parquet` (one int8 role column per
split: 0 none, 1 seed, 2 validation, 3 test). Split sizes are committed in `data/splits_report.json`,
and `make splits` refuses to overwrite splits whose sizes changed.

1. **Seed-fraction sweep** (`frac01`, `frac05`, `frac20`, `frac80`, reps `_r0`..`_r4`). NTAC's
   `at_least_one_per_class` protocol: one random seed per class, then uniform random seeds up to the
   target fraction of labelled neurons. Random streams are seeded by (task, fraction, rep).
2. **Hemisphere transfer** (`hemi_l2r`, `hemi_r2l`). Seeds are every labelled neuron on one side
   (annotation `side`); the other side is scored. The 174 `center` and 30 `na` neurons are not scored.

**Validation.** 10% of the non-seed labelled neurons are held out for tuning and early stopping; the rest
are test. Test numbers are reported once per final configuration. NTAC scores every non-seed neuron;
our test set is 90% of those, drawn at random, so its accuracy estimates the same quantity.

**Reporting.** Every number is a mean over the 5 reps (NTAC uses 5 too), broken down by super class
("region") and by cell-type size (2, 3–10, 11–100, more than 100 neurons).

**Leakage rules**

- Features come only from `features.parquet`, which `make check` proves holds no annotation column.
  Never used as features: `hemibrain_type`, `fbbt_id`/`vfb_id`, lineages, `top_nt`/`known_nt`,
  synonyms, soma or position coordinates, mirror-twin IDs, anything derived from `cell_type`.
- Partner-type features (NTAC, the partner-profile model, label-as-input GNNs) may use seed labels only.
  Non-seed neurons enter only through the model's own predictions.
- `side` is used to define the hemisphere splits, never as a feature. Mirroring neuropil columns to
  ipsi/contra uses each neuron's dominant side computed from its own synapse counts.

## Consequences

- **At the cell-type level, 1% and 5% are the same split size.** One seed per type is already 8,840
  seeds (6.4% of labelled neurons), above both targets, so both are exactly one seed per type. This
  is the hardest realistic setting: a type of 8,452 neurons (R1-6) is matched against one example.
  NTAC's own subsets are small enough that 2% leaves extra seeds for the big types.
- **120 single-neuron types** always have their only neuron as a seed and are never tested.
- **Hemisphere transfer has unseen types:** 127 types have no left-side neuron, so 107 of 61,726
  right-side test neurons (l2r) carry a type no model can predict. They stay in the test set.
- **Class-level results cover mostly optic and sensory neurons**: 68% of central-brain neurons have
  no class. Each class-level result is reported with its coverage.

## Alternatives considered

- **Stratified seeds** (seed share equal in every class). Kinder to big classes, but not NTAC's protocol,
  so our numbers would no longer sit next to theirs.
- **Uniform seeds without the per-class guarantee.** Many types would have no seed at low fractions and
  could never be predicted. NTAC reports this only as a robustness check; we may add it later the same way.
- **Validation from seeds** (cross-validation inside the labelled set). Closer to a real labelling
  workflow, but it shrinks the seed set that graph methods depend on, at exactly the low fractions that matter.
