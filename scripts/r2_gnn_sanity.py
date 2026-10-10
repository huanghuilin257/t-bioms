"""Documentation."""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore")

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import yaml                                          # noqa: E402
from scipy.stats import spearmanr                    # noqa: E402

from pparbench.gnn import GNN_NAMES, build_gnn        # noqa: E402

TARGET = "pparg"


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "hparam_grid_r2_gnn.yaml").read_text())
    d = pd.read_csv(ROOT / "data" / TARGET / "dataset.csv")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(d))
    cut = int(0.8 * len(d))
    tr, te = idx[:cut], idx[cut:]
    smi, y = d.smiles.values, d.y.values

    print(f"{TARGET} n={len(d)} train={len(tr)} test={len(te)}", flush=True)
    print("p03 (random_holdout foldmean spearman):"
          "lightgbm 0.828 / hgb 0.824 / extratrees 0.817 / rf 0.812", flush=True)
    print("defaultconfiguration 0.6",
          flush=True)

    for name in GNN_NAMES:
        p = dict(cfg["defaults"][name])
        t0 = time.time()
        try:
            m = build_gnn(name, p, seed=0, device="cuda").fit(smi[tr], y[tr])
            pr = m.predict(smi[te])
            ok = np.isfinite(pr)
            print(f"[{name:12s}] epochs={p['epochs']:3d} {time.time() - t0:7.1f}s  "
                  f"spearman={spearmanr(y[te][ok], pr[ok]).statistic:+.4f}  "
                  f"pred_sd={pr[ok].std():.3f} (y_sd={y[te].std():.3f})  "
                  f"nan={int((~ok).sum())}", flush=True)
        except Exception as exc:                      # noqa: BLE001
            import traceback
            print(f"[{name:12s}] FAIL {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=8)


if __name__ == "__main__":
    main()
