# Reproducibility package for the temporal bioactivity model-selection study

This directory contains the compact release package for the manuscript
**Model Selection against Uniform Random Selection in a 20-Target Temporal
Bioactivity Benchmark**.

For concise reference, this release package is abbreviated **T-BioMS**
(*Temporal Bioactivity Model Selection*). The full repository name and URL are
retained for discoverability and should be used when locating or citing the
repository.

This release is intentionally separate from the full research workspace. The
workspace contains exploratory analyses, intermediate predictions, historical
runs, and files that are not needed for the submitted paper. This package
contains only the paper-facing code, configuration, frozen target data, and
reproduction scripts.

Obsolete exploratory scripts, post hoc analyses not reported in the manuscript,
profiling utilities, and regenerable feature caches are excluded. The retained
supplementary scripts correspond to analyses or audits described in the paper
or Supporting Information.

## Contents

- `configs/`: protocol, model settings, and the target configuration files;
- `src/`: the reusable `pparbench` analysis library;
- `pipelines/`: the ordered curation, feature, evaluation, bootstrap, and
  reporting stages;
- `scripts/`: selected analysis and result-processing scripts; exploratory
  figure-drawing scripts are not included;
- `data/`: curated per-target tables used by the final analysis;
- `scripts/reproduce_one_target.sh`: reproduction entry point starting from a
  frozen curated table.

## Scope

This is a release package for the manuscript, not a copy of the entire
historical workspace. Large prediction files, raw external datasets, temporary
files, manuscript DOCX files, and obsolete experiments are intentionally
omitted. Final manuscript figures are maintained in the separate submission
package; figure-drawing scripts are not part of this release.

GNN checkpoints are not included in this release; all reported numerical
results were generated from the recorded configurations and random seeds.

## Data provenance

The activity data were curated from ChEMBL release 37. The target list,
endpoint choices, temporal cut-points, and curation rules are recorded in
`configs/manifests/`, `configs/targets/`, and the manuscript's Supporting
Information. The main manifest contains the 19 expansion targets;
the main PPARγ target is maintained separately in
`configs/targets/pparg.yaml` and `data/pparg/`. Together, these files define
the 20 targets used in the manuscript. Frozen molecule-level tables are
included under `data/`; raw ChEMBL exports are not redistributed here.
Reproduction therefore starts from the frozen tables rather than rerunning raw
data curation.

## Current status

The paper-facing release package is assembled and independently audited. One
complete clean-environment reproduction (`ppard`) has been run successfully.
Manuscript result snapshots
are maintained separately under `../frozen_results/`; the release audit did
not rerun all 20 targets end to end. Final manuscript figures are also
maintained outside this code-and-data release.

## Reproduction from frozen tables

Use Python 3.8 or newer to create an environment from `requirements.txt`, then
run one target from its frozen curated table:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/reproduce_one_target.sh egfr
```

The graph-network analyses require the additional packages in
`requirements-gnn.txt`. The release starts at `pipelines/p02_features.py`;
`pipelines/p01_curate.py` requires raw ChEMBL exports that are not included.

## Licenses

The code in this repository is released under the MIT License; see
`LICENSE`. The released curated tables, frozen results, and other derived
data are released under CC BY 4.0; see `LICENSE-DATA`. Third-party materials
remain subject to their own licenses and attribution requirements.

## Before public release

- Confirm that the manuscript's Code/Data Availability statement uses the
  final GitHub URL or an anonymous review link as appropriate.
- A complete clean-environment run for `ppard` is recorded in the local release
  audit notes outside this public package.
