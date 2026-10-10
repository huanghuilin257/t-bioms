"""Documentation."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")   


DESCRIPTOR_FUNCS: dict[str, callable] = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "NumHDonors": rdMolDescriptors.CalcNumHBD,
    "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
}


def _morgan_generator(radius: int, n_bits: int, chirality: bool):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=chirality
    )


def morgan_matrix(
    smiles: list[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    chirality: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Documentation."""
    gen = _morgan_generator(radius, n_bits, chirality)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    ok = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        out[i] = arr
        ok[i] = True
    return out, ok


def descriptor_matrix(smiles: list[str], names: list[str] | None = None) -> np.ndarray:
    """Documentation."""
    names = names or list(DESCRIPTOR_FUNCS)
    unknown = [n for n in names if n not in DESCRIPTOR_FUNCS]
    if unknown:
        raise ValueError(f"unknown descriptor: {unknown} available: {list(DESCRIPTOR_FUNCS)}")
    out = np.zeros((len(smiles), len(names)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        vals = []
        for n in names:
            try:
                vals.append(float(DESCRIPTOR_FUNCS[n](mol)))
            except Exception:
                vals.append(0.0)
        out[i] = np.nan_to_num(np.asarray(vals, dtype=np.float32),
                               nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_ligand_features(
    smiles: list[str], feat_cfg: dict
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Documentation."""
    fp, ok = morgan_matrix(
        smiles,
        radius=int(feat_cfg["morgan_radius"]),
        n_bits=int(feat_cfg["morgan_bits"]),
        chirality=bool(feat_cfg.get("morgan_chirality", True)),
    )
    desc_names = list(feat_cfg["descriptors"])
    desc = descriptor_matrix(smiles, desc_names)
    X = np.hstack([fp, desc]).astype(np.float32)
    names = [f"fp_{i}" for i in range(fp.shape[1])] + [f"desc_{n}" for n in desc_names]
    return X, ok, names


def build_ad_fingerprints(smiles: list[str], ad_cfg: dict) -> list:
    """Documentation."""
    gen = _morgan_generator(
        int(ad_cfg["fp_radius"]), int(ad_cfg["fp_bits"]), bool(ad_cfg.get("fp_chirality", False))
    )
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    return fps


def murcko_scaffolds(smiles: list[str]) -> list[str]:
    """Documentation."""
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            out.append("Invalid")
            continue
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            out.append("Invalid")
    return out


def canonical_smiles(smiles: list[str]) -> list[str | None]:
    """Documentation."""
    out: list[str | None] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        out.append(Chem.MolToSmiles(mol) if mol is not None else None)
    return out
