#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits      # noqa: E402

apply_thread_limits()

import numpy as np             # noqa: E402
import pandas as pd            # noqa: E402

DECLARED_IN = "manuscript protocol declaration"
GNN_MODELS = ["attentivefp", "dmpnn", "gin"]
PROTOCOLS = ["random_holdout", "scaffold_cv", "temporal"]

COLLAPSE_FRAC = 0.5      
STRAWMAN_SPEARMAN = 0.15  
N_SEEDS = 3
EXPECT_ROWS = 2592
EXPECT_CELLS = 60
EXPECT_GROUPS = 180


def targets() -> list[str]:
    """Documentation."""
    man = pd.read_csv(ROOT / "configs" / "manifests" / "targets_manifest.csv", comment="#")
    slugs = sorted(set(man["slug"]) | {"pparg"})
    return slugs


def load(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    pf = sorted(d.glob("r2_per_fold_*.csv"))
    pr = sorted(d.glob("r2_predictions_*.csv"))
    sk = sorted(d.glob("r2_skipped_*.csv"))
    if not pf:
        raise SystemExit(f"[health] {d} no r2_per_fold_*.csv")
    per_fold = pd.concat([pd.read_csv(p) for p in pf], ignore_index=True)
    preds = pd.concat([pd.read_csv(p) for p in pr], ignore_index=True)
    return per_fold, preds, sk



def check_completeness(pf: pd.DataFrame, skipped: list[Path],
                       slugs: list[str]) -> tuple[bool, list[str]]:
    msg, ok = [], True
    if skipped:
        ok = False
        msg.append(f"{len(skipped)} r2_skipped_*"
                   f"{[p.name for p in skipped]}")
    n_cells = pf.groupby(["model", "target"]).ngroups
    n_groups = pf.groupby(["model", "target", "protocol"]).ngroups
    for name, got, want in (("message", len(pf), EXPECT_ROWS),
                            ("(model,target)", n_cells, EXPECT_CELLS),
                            ("(model,target,protocol)", n_groups, EXPECT_GROUPS)):
        if got != want:
            ok = False
            msg.append(f"{name} = {got} {want}")
    missing = sorted(set(slugs) - set(pf["target"]))
    extra = sorted(set(pf["target"]) - set(slugs))
    if missing:
        ok = False
        msg.append(f"target skip {missing}")
    if extra:
        ok = False
        msg.append(f"target {extra}")
    bad_models = sorted(set(pf["model"]) ^ set(GNN_MODELS))
    if bad_models:
        ok = False
        msg.append(f"model {sorted(set(pf['model']))}")
    
    w = pf.groupby(["target", "model"]).size().unstack()
    disagree = w[w.nunique(axis=1) > 1]
    if len(disagree):
        ok = False
        msg.append(f"targetmodelinconsistent bug \n{disagree.to_string()}")
    else:
        msg.append(f"targetmodelconsistent"
                   f"{dict(w.iloc[:, 0].value_counts().sort_index())}")
    return ok, msg


def check_nan(pf: pd.DataFrame, preds: pd.DataFrame) -> tuple[bool, list[str]]:
    msg, ok = [], True
    n_nan = int(pf["n_pred_nan"].sum())
    if n_nan:
        ok = False
        bad = pf[pf.n_pred_nan > 0][["target", "protocol", "model", "fold", "seed",
                                     "n_pred_nan"]]
        msg.append(f"prediction NaN {n_nan} \n{bad.to_string(index=False)}")
    mism = pf[pf["n"] != pf["n_test"]]
    if len(mism):
        ok = False
        msg.append(f"{len(mism)} n != n_test samplestest data")
    
    
    
    hard_cols = ["rmse", "pred_std", "enrich5", "enrich10"]
    nn = pf[hard_cols].isna().sum()
    if nn.any():
        ok = False
        msg.append(f"NaN metrics NaN {dict(nn[nn > 0])}")
    for col in ["spearman", "pearson"]:
        bad_nan = pf[pf[col].isna() & (pf["pred_std"] != 0)]
        if len(bad_nan):
            ok = False
            msg.append(f"{col} {len(bad_nan)} pred_std!=0 NaN prediction")
    n_const = int((pf["pred_std"] == 0).sum())
    n_sp_nan = int(pf["spearman"].isna().sum())
    msg.append(f"pred_std==0 {n_const} spearman NaN {n_sp_nan}"
               f"{'consistent NaN prediction' if n_const == n_sp_nan else 'inconsistent '}")
    exp, got = int(pf["n"].sum()), len(preds)
    if exp != got:
        ok = False
        msg.append(f"prediction {got} ≠ Σn {exp}")
    else:
        msg.append(f"prediction = Σn = {got} NaN")
    if preds[["y_true", "y_pred"]].isna().any().any():
        ok = False
        msg.append("prediction y_true/y_pred NaN")
    return ok, msg


def check_collapse(pf: pd.DataFrame, preds: pd.DataFrame) -> tuple[bool, list[str]]:
    """Documentation."""
    ystd = (preds.groupby(["target", "protocol", "fold"])["y_true"]
            .std(ddof=1).rename("y_std").reset_index())
    m = pf.merge(ystd, on=["target", "protocol", "fold"], how="left")
    if m["y_std"].isna().any():
        return False, ["predictionmissing y_true y_std"]
    m["collapsed"] = m["pred_std"] < COLLAPSE_FRAC * m["y_std"]
    bad = m[m["collapsed"]]
    msg = [f"pred_std < {COLLAPSE_FRAC}×y_std"
           f"{len(bad)} / {len(m)} ·"]
    if len(bad):
        by = (bad.groupby(["model", "protocol"]).size()
              .rename("n_collapsed").reset_index())
        msg.append("(model,protocol) \n" + by.to_string(index=False))
        msg.append("target" + str(sorted(bad["target"].unique())))
        
        
        msg.append("target / >0 \n"
                   + m.groupby("target")["collapsed"].agg(["sum", "count"])
                     .query("sum > 0").to_string())
        m["rmse_over_ystd"] = m["rmse"] / m["y_std"]
        q = (m.groupby("collapsed")["rmse_over_ystd"]
             .describe()[["count", "25%", "50%", "75%"]].round(3))
        msg.append("rmse/y_std ≈1.000 predictionmean \n"
                   + q.to_string())
        msg.append("vs n_train median \n"
                   + m.groupby("collapsed")["n_train"].median().round(1).to_string())
        
        key = ["target", "protocol", "fold"]
        cf = set(map(tuple, bad[key].drop_duplicates().to_numpy()))
        for mdl in sorted(m["model"].unique()):
            sub = m[m["model"] == mdl].copy()
            sub["same_fold"] = [tuple(r) in cf for r in sub[key].to_numpy()]
            s = sub[sub["same_fold"]]
            if len(s):
                msg.append(f"{mdl} {int(s['collapsed'].sum())}/{len(s)}")
    return (len(bad) == 0), msg


def _agg_seed_then_fold(pf: pd.DataFrame, col: str) -> pd.DataFrame:
    """Documentation."""
    per_fold = (pf.groupby(["model", "target", "protocol", "fold"])[col]
                .median().reset_index())
    return (per_fold.groupby(["model", "target", "protocol"])[col]
            .mean().reset_index())


def check_strawman(pf: pd.DataFrame) -> tuple[bool, list[str]]:
    fm = _agg_seed_then_fold(pf, "spearman")
    bad = fm[fm["spearman"] < STRAWMAN_SPEARMAN]
    msg = [f"spearman < {STRAWMAN_SPEARMAN}"
           f"{len(bad)} / {len(fm)} (model,target,protocol)"]
    piv = fm.pivot_table(index="model", columns="protocol", values="spearman")
    msg.append("spearman targetmean \n"
               + piv.reindex(columns=PROTOCOLS).round(4).to_string())
    if len(bad):
        msg.append("\n" + bad.round(4).to_string(index=False))
    return (len(bad) == 0), msg


def check_seeds(pf: pd.DataFrame) -> tuple[bool, list[str]]:
    msg, ok = [], True
    cnt = pf.groupby(["model", "target", "protocol", "fold"]).size()
    bad = cnt[cnt != N_SEEDS]
    if len(bad):
        ok = False
        msg.append(f"{len(bad)} (model,target,protocol,) ≠ {N_SEEDS}"
                   f"{dict(bad.value_counts())}")
    else:
        msg.append(f"{N_SEEDS} {len(cnt)}")
    
    
    
    grp = pf.groupby(["model", "target", "protocol", "fold"])["spearman"]
    n_ok = grp.agg(lambda s: int(s.notna().sum()))
    spread = grp.agg(lambda s: float(s.max() - s.min()))
    assessable = n_ok >= 2
    degen = spread[assessable & (spread <= 0)]
    if len(degen):
        ok = False
        msg.append(f"{len(degen)} ≥2 NaN spearman ⇒"
                   f"\n{degen.to_string()}")
    else:
        s = spread[assessable]
        msg.append(f"{int(assessable.sum())} spearman"
                   f"{s.median():.4f} p90 {s.quantile(.9):.4f} max {s.max():.4f}")
    n_un = int((~assessable).sum())
    if n_un:
        msg.append(f"{n_un} NaN 2"
                   f"prediction check_collapse")
    return ok, msg


def check_split_parity(pf: pd.DataFrame) -> tuple[bool, list[str]]:
    """Documentation."""
    msg, ok, missing = [], True, []
    gnn = (pf[["target", "protocol", "fold", "n_train", "n"]]
           .drop_duplicates().set_index(["target", "protocol", "fold"]))
    rows = []
    for slug in sorted(pf["target"].unique()):
        p = ROOT / "results" / slug / "p03_protocol_matrix" / "per_fold.csv"
        if not p.exists():
            missing.append(slug)
            continue
        t = pd.read_csv(p)
        t = t[["protocol", "fold", "n_train", "n"]].drop_duplicates()
        t["target"] = slug
        rows.append(t)
    if missing:
        ok = False
        msg.append(f"targetno p03 per_fold.csv {missing}")
    if not rows:
        return False, msg + ["target"]
    tree = pd.concat(rows, ignore_index=True).set_index(["target", "protocol", "fold"])

    common = gnn.index.intersection(tree.index)
    only_gnn = gnn.index.difference(tree.index)
    only_tree = tree.index.difference(gnn.index)
    if len(only_gnn):
        ok = False
        msg.append(f"{len(only_gnn)} GNN {list(only_gnn)[:8]}")
    if len(only_tree):
        
        ok = False
        msg.append(f"{len(only_tree)} GNN"
                   f"{list(only_tree)[:8]}")
    diff = []
    for col in ("n_train", "n"):
        d = gnn.loc[common, col].astype(int) != tree.loc[common, col].astype(int)
        if d.any():
            diff.extend([(k, col, int(gnn.loc[k, col]), int(tree.loc[k, col]))
                         for k in common[d]])
    if diff:
        ok = False
        msg.append(f"{len(diff)} inconsistent ** bug ** {diff[:8]}")
    else:
        msg.append(f"{len(common)} (target,protocol,) n_train/n"
                   f"⇒ mixed7")
    return ok, msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/r2_gnn_v2")
    args = ap.parse_args()
    d = ROOT / args.dir
    pf, preds, skipped = load(d)
    slugs = targets()

    print(f"[health] : {DECLARED_IN}")
    print(f"[health] {args.dir}: per_fold {len(pf)} , prediction {len(preds)} ,"
          f"target {len(slugs)} manifest 19 + pparg \n")

    checks = [
        ("1 completeness", lambda: check_completeness(pf, skipped, slugs)),
        ("2 nan", lambda: check_nan(pf, preds)),
        ("3 collapse", lambda: check_collapse(pf, preds)),
        ("4 strawman", lambda: check_strawman(pf)),
        ("5 seeds", lambda: check_seeds(pf)),
        ("6 split_parity", lambda: check_split_parity(pf)),
    ]
    failed = []
    for name, fn in checks:
        ok, msg = fn()
        tag = "message" if ok else "****"
        print(f"=== {name}: {tag} ===")
        for m in msg:
            print(f"  {m}")
        print()
        if not ok:
            failed.append(name)

    if failed:
        print(f"[health] {len(failed)} {failed}")
        print("[health] 3/4 ⇒ GNN"
              "1/2/5/6 ⇒ bug")
        return 1
    print("[health]")
    print("[health] CUBLAS_WORKSPACE_CONFIG=:4096:8 +"
          "use_deterministic_algorithms + Trainer(deterministic=True)"
          "gin +89% note_8/15 12")
    print("[health] model"
          "13")
    print("[health] v2 methods"
          "mixed7 14")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
