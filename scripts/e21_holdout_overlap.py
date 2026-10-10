#!/usr/bin/env python3
"""Documentation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pparbench.threads import apply_thread_limits      # noqa: E402

apply_thread_limits()

import pandas as pd            # noqa: E402

from pparbench.config import load_protocol                          # noqa: E402
from pparbench.io import canonical_dataset_path, save_table, write_run_meta  # noqa: E402

DECLARED_IN = "manuscript protocol declaration"

HARD = 0.30        
MARK = 0.10        
CLUSTERS = ROOT / "configs" / "manifests" / "target_clusters.csv"


def mols(slug: str) -> set[str] | None:
    """Documentation."""
    p = canonical_dataset_path(slug)
    if not p.exists():
        return None
    return set(pd.read_csv(p)["smiles"].astype(str))


def main() -> int:
    cl = pd.read_csv(CLUSTERS, comment="#")
    frozen = cl.loc[cl.role == "frozen20", "slug"].tolist()
    cands = cl.loc[cl.role.isin(["holdout_batch1", "holdout_batch2"]), "slug"].tolist()

    fm, missing_f = {}, []
    for s in frozen:
        m = mols(s)
        (fm.__setitem__(s, m) if m else missing_f.append(s))
    if missing_f:
        print(f"[e21] !! FROZEN20 dataset.csv {missing_f}")
        return 1
    union = set().union(*fm.values())
    print(f"[e21] FROZEN20 {len(fm)} {len(union)} canonical SMILES")

    rows, absent = [], []
    for s in cands:
        m = mols(s)
        if not m:
            absent.append(s)
            continue
        inter = m & union
        per = {f: len(m & v) / len(m) for f, v in fm.items()}
        top = max(per, key=per.get)
        rows.append({
            "slug": s, "role": cl.set_index("slug").loc[s, "role"],
            "cluster_fine": cl.set_index("slug").loc[s, "cluster_fine"],
            "n_mols": len(m), "n_shared_union": len(inter),
            "overlap_union": round(len(inter) / len(m), 4),
            "overlap_max_single": round(per[top], 4), "most_similar_frozen": top,
            "verdict": ("message" if len(inter) / len(m) > HARD
                        else "message" if len(inter) / len(m) > MARK else "message"),
        })
    out = pd.DataFrame(rows).sort_values("overlap_union", ascending=False)

    
    inner = []
    for a in frozen:
        for b in frozen:
            if a >= b:
                continue
            n = len(fm[a] & fm[b])
            inner.append({"a": a, "b": b, "n_shared": n,
                          "overlap_a": round(n / len(fm[a]), 4),
                          "overlap_b": round(n / len(fm[b]), 4),
                          "overlap_sym": round(n / min(len(fm[a]), len(fm[b])), 4)})
    inner = pd.DataFrame(inner).sort_values("overlap_sym", ascending=False)

    d = ROOT / "results" / "cross_target" / "e21_holdout_overlap"
    d.mkdir(parents=True, exist_ok=True)
    save_table(out, d / "e21_candidate_overlap.csv")
    save_table(inner, d / "e21_frozen20_inner_overlap.csv")

    pd.set_option("display.width", 200)
    print(f"\n[e21] {len(out)} dataset.csv {absent or ''}")
    print(out.to_string(index=False))
    print(f"\n[e21] {(out.verdict=='').sum()} /"
          f"{(out.verdict=='').sum()} / {(out.verdict=='').sum()}")
    print("\n[e21] FROZEN20 10")
    print(inner.head(10).to_string(index=False))

    write_run_meta(d, target="cross_target", stage="e21_holdout_overlap",
                   protocol_fingerprint=load_protocol().fingerprint,
                   extra={"declared_in": DECLARED_IN, "hard_threshold": HARD,
                          "mark_threshold": MARK, "n_frozen": len(fm),
                          "n_union_smiles": len(union), "absent": absent})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
