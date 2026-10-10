from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pparbench.config import load_protocol, load_target
from pparbench.io import save_table, target_results_dir, write_run_meta
from pparbench.metrics import bootstrap_ci, paired_bootstrap
from pparbench.models import FAMILY_CONTRAST, family_of


def main() -> int:
    ap = argparse.ArgumentParser(description="bootstrap CI")
    ap.add_argument("--target", required=True)
    ap.add_argument("--n-boot", type=int, default=None, help="protocol.yaml n_boot")
    args = ap.parse_args()

    prot = load_protocol()
    tcfg = load_target(args.target)

    pred_path = target_results_dir(tcfg.slug, "p03_protocol_matrix") / "predictions.csv"
    if not pred_path.exists():
        print(f"[error] missing {pred_path} run first p03_protocol_matrix.py --target {tcfg.slug}")
        return 1
    pred = pd.read_csv(pred_path)

    n_boot = args.n_boot or int(prot.bootstrap["n_boot"])
    ci = float(prot.bootstrap["ci"])
    seed = prot.seed
    print(f"[p04] target={tcfg.slug}  n_boot={n_boot}  ci={ci}  seed={seed}")

    
    ci_rows = []
    for (protocol, fold, model), g in pred.groupby(["protocol", "fold", "model"]):
        r = bootstrap_ci(g["y_true"].to_numpy(), g["y_pred"].to_numpy(),
                         n_boot=n_boot, ci=ci, seed=seed)
        ci_rows.append({"protocol": protocol, "fold": fold, "model": model,
                        "family": family_of(model), "n_test": len(g),
                        "ci_width": r["ci_high"] - r["ci_low"], **r})
    ci_df = pd.DataFrame(ci_rows)
    outdir = target_results_dir(tcfg.slug, "p04_bootstrap")
    save_table(ci_df, outdir / "model_ci.csv")

    print(f"[p04] model CI: {len(ci_df)}")
    w = ci_df.groupby("protocol")["ci_width"].agg(["mean", "max"])
    print("[p04] CI samples fold")
    print(w.to_string())
    
    best = ci_df.loc[ci_df["ci_high"].idxmax()]
    print(f"[p04] CI : {best['protocol']}/{best['fold']}/{best['model']}"
          f"= {best['ci_high']:.3f}")

    
    pair_rows = []
    for (protocol, fold), g in pred.groupby(["protocol", "fold"]):
        
        wide = g.pivot_table(index="mol_id", columns="model",
                             values="y_pred", aggfunc="first")
        truth = g.groupby("mol_id")["y_true"].first().reindex(wide.index)
        wide = wide.dropna()
        truth = truth.reindex(wide.index)
        for a, b in itertools.combinations(sorted(wide.columns), 2):
            r = paired_bootstrap(truth.to_numpy(), wide[a].to_numpy(), wide[b].to_numpy(),
                                 n_boot=n_boot, ci=ci, seed=seed)
            pair_rows.append({
                "protocol": protocol, "fold": fold,
                "model_a": a, "model_b": b,
                "family_a": family_of(a), "family_b": family_of(b),
                "n_test": len(wide), **r,
            })
    pair_df = pd.DataFrame(pair_rows)
    save_table(pair_df, outdir / "pairwise.csv")
    n_sig = int(pair_df["significant"].sum())
    print(f"\n[p04] : {len(pair_df)} {n_sig}"
          f"({n_sig/max(1,len(pair_df)):.0%})")

    # ---------------- 3. bagging vs boosting ----------------
    
    #
    
    
    
    
    cross = pair_df[
        pair_df["family_a"].isin(FAMILY_CONTRAST)
        & pair_df["family_b"].isin(FAMILY_CONTRAST)
        & (pair_df["family_a"] != pair_df["family_b"])
    ].copy()
    
    flip = cross["family_a"] == "boosting"
    cross["delta_bag_minus_boost"] = np.where(flip, -cross["delta"], cross["delta"])
    cross["bagging_model"] = np.where(flip, cross["model_b"], cross["model_a"])
    cross["boosting_model"] = np.where(flip, cross["model_a"], cross["model_b"])

    fam_rows = []
    for protocol, g in cross.groupby("protocol"):
        d = g["delta_bag_minus_boost"]
        fam_rows.append({
            "protocol": protocol,
            "n_comparisons": len(g),
            "bagging_wins": int((d > 0).sum()),
            "bagging_win_rate": float((d > 0).mean()),
            "mean_delta": float(d.mean()),
            "median_delta": float(d.median()),
            "n_significant": int(g["significant"].sum()),
            "n_significant_favoring_bagging": int(((d > 0) & g["significant"]).sum()),
        })
    fam_df = pd.DataFrame(fam_rows).sort_values("protocol")
    save_table(fam_df, outdir / "family_contrast.csv")
    print("\n[p04] bagging − boosting = bagging")
    print(fam_df.to_string(index=False))

    if len(fam_df) > 1:
        signs = {r["protocol"]: np.sign(r["mean_delta"]) for _, r in fam_df.iterrows()}
        if len(set(signs.values())) > 1:
            print("⇒ ****protocol ②")
        else:
            print("⇒ consistent; differences are in magnitude protocol")

    
    
    
    
    
    
    decay_df = None
    if {"random_holdout", "temporal"} <= set(ci_df["protocol"]):
        wide = (ci_df.groupby(["model", "protocol"])["spearman"].mean()
                .unstack("protocol"))
        if {"random_holdout", "temporal"} <= set(wide.columns):
            decay_df = pd.DataFrame({
                "model": wide.index,
                "family": [family_of(m) for m in wide.index],
                "random_holdout": wide["random_holdout"].round(4),
                "temporal": wide["temporal"].round(4),
                "decay": (wide["random_holdout"] - wide["temporal"]).round(4),
            }).sort_values("decay", ascending=False).reset_index(drop=True)
            save_table(decay_df, outdir / "model_decay.csv")
            print("\n[p04] model random temporal")
            print(decay_df.to_string(index=False))
            worst = decay_df.iloc[0]["model"]
            if "knn" in decay_df["model"].values:
                rank = int(decay_df.index[decay_df["model"] == "knn"][0]) + 1
                print(f"kNN {rank}/{len(decay_df)}"
                      f"{worst}")
                if rank == 1:
                    print("⇒ supported model")
                else:
                    print("⇒ **supported** kNN"
                          "protocol")

    write_run_meta(
        outdir, target=tcfg.slug, stage="p04_bootstrap",
        protocol_fingerprint=prot.fingerprint,
        extra={"n_boot": n_boot, "ci": ci,
               "n_model_ci": len(ci_df), "n_pairwise": len(pair_df),
               "n_significant_pairwise": n_sig,
               "max_ci_high": float(ci_df["ci_high"].max())},
    )
    print(f"\n[p04] results {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
