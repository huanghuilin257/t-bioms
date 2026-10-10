#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))




from pparbench.threads import apply_thread_limits            # noqa: E402

apply_thread_limits(1)

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import yaml                                                   # noqa: E402
from rdkit import Chem, DataStructs, RDLogger                 # noqa: E402
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator      # noqa: E402

RDLogger.DisableLog("rdApp.*")

from pparbench.config import CONFIG_DIR, DATA_DIR, RESULTS_DIR  # noqa: E402




GUO_DIR = ROOT.parents[1] / "external" / "UMAP_split_for_VS-main"
GUO_CSV = GUO_DIR / "data" / "clustering_id_k7.csv"


GUO_TARGETS = {"random": 0.717, "scaffold": 0.625, "butina": 0.417, "umap": 0.375}


GUO_SPLIT_COLS = {
    "umap": "Cluster_ID",
    "scaffold": "Scaffold_Cluster_ID",
    "random": "Random_Cluster_ID",
    "butina": "Butina_Cluster_ID",
}

OUT_DIR = RESULTS_DIR / "e11_split_difficulty"





FINGERPRINTS: tuple[str, ...] = (
    "morgan_r2_2048",         
    "morgan_r2_1024",         
    "morgan_r3_2048",         
    "morgan_r1_2048",         
    "morgan_r2_2048_count",   
    "featmorgan_r2_2048",     
    "rdkit_2048",             
    "atompair_2048",          
    "topotorsion_2048",       
    "maccs_167",              
)

TRAIN_VARIANTS = ("all_other_6", "exclude_val_5")
AGGREGATIONS = ("fold_mean", "pooled")

TOL_HIT = 0.010
TOL_NEAR = 0.025



def _make_fp_func(name: str):
    """Documentation."""
    if name == "maccs_167":
        return MACCSkeys.GenMACCSKeys
    if name.startswith("morgan") or name.startswith("featmorgan"):
        parts = name.split("_")
        radius = int(parts[1][1:])
        size = int(parts[2])
        kw = {"radius": radius, "fpSize": size}
        if name.startswith("featmorgan"):
            kw["atomInvariantsGenerator"] = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        gen = rdFingerprintGenerator.GetMorganGenerator(**kw)
        if name.endswith("_count"):
            return gen.GetCountFingerprint
        return gen.GetFingerprint
    if name.startswith("rdkit_"):
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=int(name.split("_")[1]))
        return gen.GetFingerprint
    if name.startswith("atompair_"):
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=int(name.split("_")[1]))
        return gen.GetFingerprint
    if name.startswith("topotorsion_"):
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=int(name.split("_")[1]))
        return gen.GetFingerprint
    raise ValueError(f"unknown {name}")


def build_fps(smiles: list[str], fp_name: str) -> tuple[list, np.ndarray]:
    """Documentation."""
    fn = _make_fp_func(fp_name)
    fps: list = []
    ok = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            fps.append(None)
            continue
        try:
            fps.append(fn(mol))
            ok[i] = True
        except Exception:
            fps.append(None)
    return fps, ok



#





#





def _pack_bitvect_matrix(fps: list) -> tuple[np.ndarray, np.ndarray]:
    """Documentation."""
    raw = [np.frombuffer(DataStructs.BitVectToBinaryText(fp), dtype=np.uint8) for fp in fps]
    nbytes = raw[0].size
    pad = (-nbytes) % 8
    mat = np.zeros((len(raw), nbytes + pad), dtype=np.uint8)
    for i, r in enumerate(raw):
        mat[i, :nbytes] = r
    packed = np.ascontiguousarray(mat).view(np.uint64)
    pc = np.bitwise_count(packed).sum(axis=1, dtype=np.int64)
    return packed, pc


def _max_sim_bits(test_fps: list, train_fps: list, chunk: int = 8) -> np.ndarray:
    A, pc_a = _pack_bitvect_matrix(train_fps)
    T, pc_t = _pack_bitvect_matrix(test_fps)
    out = np.empty(len(test_fps), dtype=np.float64)
    for s in range(0, T.shape[0], chunk):
        t = T[s:s + chunk]
        inter = np.bitwise_count(t[:, None, :] & A[None, :, :]).sum(axis=-1, dtype=np.int32)
        union = pc_t[s:s + chunk, None] + pc_a[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        out[s:s + chunk] = sim.max(axis=1)
    return out


def _csr_from_count_fps(fps: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Documentation."""
    counts = [fp.GetNonzeroElements() for fp in fps]
    nnz = np.array([len(c) for c in counts], dtype=np.int64)
    data = np.fromiter((v for c in counts for v in c.values()), dtype=np.int64,
                       count=int(nnz.sum()))
    indices = np.fromiter((k for c in counts for k in c.keys()), dtype=np.int64,
                          count=int(nnz.sum()))
    row_of_nnz = np.repeat(np.arange(len(fps), dtype=np.int64), nnz)
    row_sum = np.bincount(row_of_nnz, weights=data, minlength=len(fps))
    return data, indices, row_of_nnz, row_sum


def _max_sim_counts(test_fps: list, train_fps: list) -> np.ndarray:
    a_data, a_idx, a_row, a_sum = _csr_from_count_fps(train_fps)
    m = len(train_fps)
    length = max(int(a_idx.max()) + 1, 1)
    t_counts = [fp.GetNonzeroElements() for fp in test_fps]
    length = max(length, max((max(c) + 1 for c in t_counts if c), default=1))
    dense = np.zeros(length, dtype=np.int64)

    out = np.empty(len(test_fps), dtype=np.float64)
    for i, c in enumerate(t_counts):
        if not c:
            out[i] = 0.0
            continue
        idx = np.fromiter(c.keys(), dtype=np.int64, count=len(c))
        val = np.fromiter(c.values(), dtype=np.int64, count=len(c))
        dense[idx] = val
        inter = np.bincount(a_row, weights=np.minimum(a_data, dense[a_idx]), minlength=m)
        union = a_sum + val.sum() - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        out[i] = float(sim.max())
        dense[idx] = 0          
    return out


def max_sim_to_train(test_fps: list, train_fps: list, *, count_based: bool) -> np.ndarray:
    """Documentation."""
    if not train_fps or not test_fps:
        return np.full(len(test_fps), np.nan)
    if count_based:
        return _max_sim_counts(test_fps, train_fps)
    return _max_sim_bits(test_fps, train_fps)



def _val_fold_of(test_fold: int, n_folds: int = 7) -> int:
    """Documentation."""
    return n_folds if test_fold == 1 else test_fold - 1


def calibrate_task(fp_name: str, split: str, variant: str, limit: int | None) -> list[dict]:
    """Documentation."""
    col = GUO_SPLIT_COLS[split]
    df = pd.read_csv(GUO_CSV)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)

    fps, ok = build_fps(df["SMILES"].tolist(), fp_name)
    n_bad = int((~ok).sum())
    df = df.loc[ok].reset_index(drop=True)
    fps = [f for f, k in zip(fps, ok) if k]

    folds = df[col].to_numpy()
    uniq = sorted(set(folds.tolist()))
    per_fold_mean: list[float] = []
    per_fold_n: list[int] = []
    pooled_vals: list[np.ndarray] = []
    for f in uniq:
        test_idx = np.where(folds == f)[0]
        if variant == "all_other_6":
            train_mask = folds != f
        else:
            vf = _val_fold_of(int(f), n_folds=len(uniq))
            train_mask = (folds != f) & (folds != vf)
        train_idx = np.where(train_mask)[0]
        sims = max_sim_to_train([fps[i] for i in test_idx], [fps[i] for i in train_idx],
                                count_based=fp_name.endswith('_count'))
        per_fold_mean.append(float(np.nanmean(sims)))
        per_fold_n.append(len(test_idx))
        pooled_vals.append(sims)

    all_vals = np.concatenate(pooled_vals)
    rows: list[dict] = []
    for agg in AGGREGATIONS:
        value = (
            float(np.mean(per_fold_mean))
            if agg == "fold_mean"
            else float(np.nanmean(all_vals))
        )
        rows.append(
            {
                "fingerprint": fp_name,
                "train_variant": variant,
                "aggregation": agg,
                "split": split,
                "value": round(value, 4),
                "guo_target": GUO_TARGETS[split],
                "abs_dev": round(abs(value - GUO_TARGETS[split]), 4),
                "n_mol": int(len(df)),
                "n_unparsable": n_bad,
                "fold_sizes": ";".join(str(x) for x in per_fold_n),
            }
        )
    return rows


def verdict(max_dev: float) -> str:
    if max_dev <= TOL_HIT:
        return "HIT"
    if max_dev <= TOL_NEAR:
        return "NEAR"
    return "MISS"


def step_calibrate(jobs: int, limit: int | None) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not GUO_CSV.exists():
        raise SystemExit(f"Guo fold {GUO_CSV}")

    t0 = time.time()
    tasks = [(fp, sp, va) for fp in FINGERPRINTS
             for sp in GUO_SPLIT_COLS for va in TRAIN_VARIANTS]
    print(f"{len(FINGERPRINTS)} × {len(GUO_SPLIT_COLS)} ×"
          f"{len(TRAIN_VARIANTS)} train = {len(tasks)} {jobs}")
    rows: list[dict] = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as ex:
        futs = {ex.submit(calibrate_task, fp, sp, va, limit): (fp, sp, va)
                for fp, sp, va in tasks}
        for fut in as_completed(futs):
            fp, sp, va = futs[fut]
            n_done += 1
            try:
                r = fut.result()
            except Exception as exc:                       # noqa: BLE001
                print(f"  [FAIL] {fp}/{sp}/{va}: {exc}", flush=True)
                continue
            rows.extend(r)
            print(f"  [{n_done}/{len(tasks)}] {fp}/{sp}/{va}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("FAIL reason")

    df = pd.DataFrame(rows)
    
    key = ["fingerprint", "train_variant", "aggregation"]
    dev = df.groupby(key)["abs_dev"].max().rename("max_abs_dev").reset_index()
    dev["verdict"] = dev["max_abs_dev"].map(verdict)
    df = df.merge(dev, on=key, how="left")
    df = df.sort_values(["max_abs_dev", "fingerprint", "split"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "calibration_guo.csv", index=False)

    
    order_ok = []
    for (fp, v, a), g in df.groupby(key):
        vals = g.set_index("split")["value"]
        if not set(GUO_TARGETS).issubset(vals.index):
            
            order_ok.append({"fingerprint": fp, "train_variant": v, "aggregation": a,
                             "order_reproduced": False, "incomplete": True,
                             "max_abs_dev": float(g["max_abs_dev"].iloc[0]),
                             "verdict": "INCOMPLETE"})
            continue
        ok = vals["random"] > vals["scaffold"] > vals["butina"] > vals["umap"]
        order_ok.append({"fingerprint": fp, "train_variant": v, "aggregation": a,
                         "order_reproduced": bool(ok), "incomplete": False,
                         "max_abs_dev": float(g["max_abs_dev"].iloc[0]),
                         "verdict": g["verdict"].iloc[0]})
    od = pd.DataFrame(order_ok).sort_values("max_abs_dev").reset_index(drop=True)
    od.to_csv(OUT_DIR / "calibration_order.csv", index=False)

    _write_calibration_md(df, od, limit, time.time() - t0)
    return df


def _write_calibration_md(df: pd.DataFrame, od: pd.DataFrame,
                          limit: int | None, secs: float) -> None:
    best = od.iloc[0]
    lines = [
        "# e11 Guo 2025 Fig 2b",
        "",
        f"{secs:.0f}s molecules {int(df['n_mol'].iloc[0])}"
        + (f"--limit {limit} **results**" if limit else ""),
        f"molecules {int(df['n_unparsable'].max())} consistent",
        "",
        "Guo 2025 Fig 2b random 0.717 / scaffold 0.625 / Butina 0.417 / UMAP 0.375",
        f"HIT ≤ {TOL_HIT} / NEAR ≤ {TOL_NEAR} / MISS",
        "",
        "##",
        "",
        "| | train | | | | |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in od.iterrows():
        lines.append(
            f"| {r['fingerprint']} | {r['train_variant']} | {r['aggregation']} | "
            f"{r['max_abs_dev']:.4f} | {r['verdict']} | "
            f"{'' if r['order_reproduced'] else ''} |"
        )
    lines += [
        "",
        "##",
        "",
        "| | | Guo | |",
        "|---|---|---|---|",
    ]
    g = df[(df.fingerprint == best["fingerprint"])
           & (df.train_variant == best["train_variant"])
           & (df.aggregation == best["aggregation"])]
    for split in ["random", "scaffold", "butina", "umap"]:
        r = g[g.split == split].iloc[0]
        lines.append(f"| {split} | {r['value']:.4f} | {r['guo_target']:.3f} | {r['abs_dev']:.4f} |")

    n_hit = int((od.verdict == "HIT").sum())
    n_near = int((od.verdict == "NEAR").sum())
    lines += [
        "",
        "##",
        "",
        f"- HIT {n_hit} NEAR {n_near} {len(od)}",
    ]
    if n_hit + n_near == 0:
        lines += [
            "- ** ** docstring results",
            "Fig 2b",
            "****",
            "Guo",
        ]
    else:
        lines += [
            f"- `{best['fingerprint']} / {best['train_variant']} / {best['aggregation']}`",
            "- ****",
        ]
    if bool(od["order_reproduced"].any()):
        lines.append(
            f"- random > scaffold > Butina > UMAP"
            f"{int(od['order_reproduced'].sum())}/{len(od)}"
            "******** methods"
        )
    (OUT_DIR / "calibration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def _target_slugs() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "targets").glob("*.yaml")
                  if not p.stem.startswith("_"))


def temporal_one(slug: str, fp_name: str) -> list[dict]:
    """Documentation."""
    cfg = yaml.safe_load((CONFIG_DIR / "targets" / f"{slug}.yaml").read_text(encoding="utf-8"))
    cuts = cfg.get("cutpoints") or []
    ds = DATA_DIR / slug / "dataset.csv"
    if not ds.exists() or not cuts:
        return [{"slug": slug, "cutpoint": None, "value": np.nan,
                 "note": "dataset.csv"}]

    df = pd.read_csv(ds)
    n_all = len(df)
    n_missing_year = int(df["year"].isna().sum())
    df = df.dropna(subset=["year", "smiles"]).reset_index(drop=True)

    fps, ok = build_fps(df["smiles"].tolist(), fp_name)
    n_bad = int((~ok).sum())
    df = df.loc[ok].reset_index(drop=True)
    fps = [f for f, k in zip(fps, ok) if k]
    years = df["year"].to_numpy(dtype=float)

    rows: list[dict] = []
    for T in cuts:
        tr = np.where(years < T)[0]
        te = np.where(years >= T)[0]
        sims = max_sim_to_train([fps[i] for i in te], [fps[i] for i in tr],
                                count_based=fp_name.endswith('_count'))
        rows.append({
            "slug": slug,
            "cutpoint": int(T),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "value": round(float(np.nanmean(sims)), 4) if len(te) else np.nan,
            "sd": round(float(np.nanstd(sims)), 4) if len(te) else np.nan,
            "n_all_rows": n_all,
            "n_missing_year": n_missing_year,
            "n_unparsable": n_bad,
            "fingerprint": fp_name,
            "note": "",
        })
    return rows


def step_apply(jobs: int, fp_name: str) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slugs = _target_slugs()
    print(f"{len(slugs)} target {fp_name}")
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(jobs, len(slugs))) as ex:
        futs = {ex.submit(temporal_one, s, fp_name): s for s in slugs}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:                       # noqa: BLE001
                print(f"  [FAIL] {s}: {exc}", flush=True)
                rows.append({"slug": s, "cutpoint": None, "value": np.nan,
                             "note": f"FAIL: {exc}"})
                continue
            rows.extend(r)
            print(f"  [done] {s}  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["slug", "cutpoint"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "temporal_all.csv", index=False)

    
    for slug, g in df.groupby("slug"):
        g.to_csv(OUT_DIR / f"temporal_{slug}.csv", index=False)

    ok = df.dropna(subset=["value"])
    if len(ok):
        print("\n=== target 7 mean ===")
        per_t = ok.groupby("slug")["value"].agg(["mean", "min", "max", "count"])
        print(per_t.round(4).to_string())
        print(f"\ntargetmedian {ok['value'].median():.4f}"
              f"（Guo：random 0.717 / scaffold 0.625 / Butina 0.417 / UMAP 0.375）")
    return df


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", choices=["calibrate", "apply"], required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="N molecules")
    ap.add_argument("--fingerprint", default=None,
                    help="default calibration_order.csv")
    args = ap.parse_args()

    if args.step == "calibrate":
        step_calibrate(args.jobs, args.limit)
        print(f"\nwrite {OUT_DIR}/calibration_guo.csv calibration_summary.md")
        return

    fp = args.fingerprint
    if fp is None:
        order = OUT_DIR / "calibration_order.csv"
        if not order.exists():
            raise SystemExit("run first --step calibrate --fingerprint")
        od = pd.read_csv(order)
        fp = str(od.iloc[0]["fingerprint"])
        print(f"{fp} {od.iloc[0]['verdict']}"
              f"{od.iloc[0]['max_abs_dev']:.4f}")
        if od.iloc[0]["verdict"] == "MISS":
            print("⚠ MISS **** Guo")
    step_apply(args.jobs, fp)
    print(f"\nwrite {OUT_DIR}/temporal_all.csv")


if __name__ == "__main__":
    main()
