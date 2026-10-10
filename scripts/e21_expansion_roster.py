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
import yaml                    # noqa: E402

from pparbench.config import load_protocol                          # noqa: E402
from pparbench.io import save_table, write_run_meta                 # noqa: E402

DECLARED_IN = "manuscript protocol declaration"

HARD = 0.30            
MARK = 0.10            
FULL_CUTPOINTS = 7     

CLUSTERS = ROOT / "configs" / "manifests" / "target_clusters.csv"
OVERLAP = ROOT / "results" / "cross_target" / "e21_holdout_overlap" / "e21_candidate_overlap.csv"



PRE_EXCLUDED = {
    "CHEMBL4078": "eel AChE FROZEN20 ache",
    "CHEMBL302": "rat A2a FROZEN20 adora2a",
    "CHEMBL4153": "guinea pig σ1 sigmar1",
}


def cutpoints(slug: str) -> list[int] | None:
    p = ROOT / "configs" / "targets" / f"{slug}.yaml"
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text(encoding="utf-8")).get("cutpoints")


def main() -> int:
    cl = pd.read_csv(CLUSTERS, comment="#").set_index("slug")
    if not OVERLAP.exists():
        print(f"[e21b] !! {OVERLAP.relative_to(ROOT)} run first scripts/e21_holdout_overlap.py")
        return 1
    ov = pd.read_csv(OVERLAP).set_index("slug")

    cands = cl.index[cl.role.isin(["holdout_batch1", "holdout_batch2"])].tolist()
    missing = [s for s in cands if s not in ov.index]
    if missing:
        print(f"[e21b] !! no overlap {missing}")
        return 1

    rows = []
    for s in cands:
        cp = cutpoints(s)
        n_cp = len(cp) if cp else 0
        o = float(ov.loc[s, "overlap_union"])
        reasons = []
        if n_cp == 0:
            reasons.append("test>=250 train>=500 train>=test")
        if o > HARD:
            reasons.append(f"overlap_union={o:.4f} > {HARD}")
        rows.append({
            "slug": s,
            "role": cl.loc[s, "role"],
            "cluster_fine": cl.loc[s, "cluster_fine"],
            "cluster_coarse": cl.loc[s, "cluster_coarse"],
            "n_mols": int(ov.loc[s, "n_mols"]),
            "n_cutpoints": n_cp,
            "cutpoints": "" if not cp else ";".join(str(c) for c in cp),
            "overlap_union": o,
            "overlap_max_single": float(ov.loc[s, "overlap_max_single"]),
            "most_similar_frozen": ov.loc[s, "most_similar_frozen"],
            "overlap_verdict": ov.loc[s, "verdict"],
            "admitted": not reasons,
            "reason_primary": reasons[0] if reasons else "",
            "reasons_all": " | ".join(reasons),
        })
    df = pd.DataFrame(rows)

    adm = (df[df.admitted]
           .drop(columns=["admitted", "reason_primary", "reasons_all"])
           .sort_values(["n_cutpoints", "overlap_union"], ascending=[False, True]))
    fil = (df[~df.admitted]
           .drop(columns=["admitted", "cutpoints"])
           .sort_values(["reason_primary", "overlap_union"], ascending=[True, False]))

    d = ROOT / "results" / "cross_target" / "e21_expansion_roster"
    d.mkdir(parents=True, exist_ok=True)
    save_table(adm, d / "e21_admitted.csv")
    save_table(fil, d / "e21_filtered.csv")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 60)
    print(f"[e21b] {len(df)} = batch1 {int((df.role=='holdout_batch1').sum())}"
          f" + batch2 {int((df.role=='holdout_batch2').sum())}")
    print(f"\n[e21b] {len(adm)}")
    print(adm.to_string(index=False))
    print(f"\n[e21b] {len(fil)} 97 + reason")
    print(fil[["slug", "role", "n_mols", "n_cutpoints", "overlap_union",
               "reasons_all"]].to_string(index=False))

    
    clean = adm[adm.overlap_verdict == "message"]
    print("\n[e21b] 100 89")
    for label, sub in (("+", adm), ("message", clean)):
        full = int((sub.n_cutpoints >= FULL_CUTPOINTS).sum())
        print(f"{label}: {len(sub)} {FULL_CUTPOINTS} {full}"
              f"{len(sub) - full} median {int(sub.n_cutpoints.median())}")
    print("\n[e21b] 7 target target")
    short = adm[adm.n_cutpoints < FULL_CUTPOINTS][["slug", "role", "n_cutpoints", "overlap_verdict"]]
    print(short.to_string(index=False) if len(short) else "message")

    print("\n[e21b] p01 target 92")
    for k, v in PRE_EXCLUDED.items():
        print(f"  {k}  {v}")

    write_run_meta(d, target="cross_target", stage="e21_expansion_roster",
                   protocol_fingerprint=load_protocol().fingerprint,
                   extra={"declared_in": DECLARED_IN, "hard_threshold": HARD,
                          "mark_threshold": MARK, "full_cutpoints": FULL_CUTPOINTS,
                          "n_candidates": len(df), "n_admitted": len(adm),
                          "n_filtered": len(fil), "n_clean": len(clean),
                          "pre_excluded": PRE_EXCLUDED})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
