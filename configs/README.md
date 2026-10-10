# Configuration files

- Root YAML files define the shared protocol and hyperparameter grids.
- `targets/` contains target-specific endpoint, assay, and temporal settings.
- `manifests/` contains target lists, cluster assignments, and Landrum metadata.
- `targets/pparg.yaml` and the matching frozen table define the main PPARγ target.

`manifests/targets_manifest.csv` contains the 19 expansion targets. Together
with PPARγ, these files define the 20-target analysis set.

The release package includes the frozen, already-curated tables under
`data/<target>/dataset.csv`. These are the inputs used for a reproduction run.
The original target configuration still records the ChEMBL export path used
for the historical curation step; the raw exports are not redistributed here.
Therefore, a reproduction from this package starts at `p02_features.py`, not
at `p01_curate.py`. Re-running curation from raw ChEMBL exports requires the
exports described in the manuscript and Supporting Information.

