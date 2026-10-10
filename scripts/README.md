# Script scope

The scripts in this directory are the portable analysis and audit scripts
retained for the public release. They do not all belong to the same
reproduction path.

## Primary manuscript path

- `../pipelines/p02_features.py` through `../pipelines/p06_report.py`: the
  single-target frozen-data pipeline;
- `e04*`: model-pool and selection-cost analyses;
- `e10*`, `e11*`: temporal and similarity analyses;
- `e13*`, `e14*`: paired and cluster-bootstrap analyses;
- `e17*` through `e24*`: protocol-gap, model-class, and abstention analyses;

## Supplementary or audit paths

Scripts such as `e01*`, `e06*`, `e07*`, `e08*`, `e12*`, `e15*`,
`e20*`, `e21*`, `e22*`, `e23*`, and the retained `r2_gnn_*` scripts support
sensitivity checks, provenance checks, or supplementary analyses. They are not
all required to regenerate the manuscript results, but each retained script
belongs to an analysis or audit reported in the manuscript or Supporting
Information.

The manuscript-facing mapping and audit notes are maintained with the paper
working files outside this public code-and-data package.

Supplementary scripts that read `results/` require the corresponding upstream
analysis outputs to be generated first. Manuscript result snapshots are stored
outside this code package under `../../frozen_results/` and are not a drop-in
replacement for every intermediate input.
