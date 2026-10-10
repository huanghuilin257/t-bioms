"""Documentation."""
from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np



os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/no_such_mps_dir_pafb")

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch                                              # noqa: E402
import torch.nn as nn                                     # noqa: E402
from rdkit import Chem                                    # noqa: E402
from torch_geometric.data import Data                      # noqa: E402
from torch_geometric.loader import DataLoader              # noqa: E402
from torch_geometric.nn import global_add_pool             # noqa: E402
from torch_geometric.nn.models import GIN, AttentiveFP     # noqa: E402

GNN_NAMES = ("dmpnn", "attentivefp", "gin")


def enable_determinism() -> None:
    """Documentation."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
_ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 34]     # C N O F P S Cl Br I B Si Se
_HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2]
_BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
          Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

ATOM_DIM = len(_ELEMENTS) + 1 + 6 + 5 + 5 + len(_HYB) + 1 + 2 + 1
BOND_DIM = len(_BONDS) + 1 + 2


def _onehot(v, choices) -> list[float]:
    """Documentation."""
    out = [0.0] * (len(choices) + 1)
    out[choices.index(v) if v in choices else len(choices)] = 1.0
    return out


def _atom_features(a: Chem.Atom) -> list[float]:
    return (
        _onehot(a.GetAtomicNum(), _ELEMENTS)
        + _onehot(a.GetTotalDegree(), [0, 1, 2, 3, 4])
        + _onehot(a.GetFormalCharge(), [-2, -1, 0, 1])
        + _onehot(a.GetTotalNumHs(), [0, 1, 2, 3])
        + _onehot(a.GetHybridization(), _HYB)
        + [float(a.GetIsAromatic()), float(a.IsInRing())]
        + [a.GetMass() * 0.01]
    )


def _bond_features(b: Chem.Bond) -> list[float]:
    return (_onehot(b.GetBondType(), _BONDS)
            + [float(b.GetIsConjugated()), float(b.IsInRing())])


def smiles_to_data(smi: str, y: float = 0.0) -> Data | None:
    """Documentation."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = _bond_features(b)
        src += [i, j]; dst += [j, i]; eattr += [f, f]      
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float)
    else:                                                  
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_DIM), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([y], dtype=torch.float))


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class _GinRegressor(nn.Module):
    """Documentation."""

    def __init__(self, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.gnn = GIN(in_channels=ATOM_DIM, hidden_channels=hidden_channels,
                       num_layers=num_layers, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                  nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden_channels, 1))

    def forward(self, batch) -> torch.Tensor:
        h = self.gnn(batch.x, batch.edge_index)
        return self.head(global_add_pool(h, batch.batch)).view(-1)


class _AttentiveFPRegressor(nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int,
                 num_timesteps: int, dropout: float):
        super().__init__()
        self.net = AttentiveFP(in_channels=ATOM_DIM, hidden_channels=hidden_channels,
                               out_channels=1, edge_dim=BOND_DIM,
                               num_layers=num_layers, num_timesteps=num_timesteps,
                               dropout=dropout)

    def forward(self, batch) -> torch.Tensor:
        return self.net(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)


class PyGRegressor:
    """Documentation."""

    def __init__(self, kind: str, params: dict[str, Any], *, seed: int, device: str):
        self.kind, self.p, self.seed, self.device = kind, dict(params), seed, device
        self.model: nn.Module | None = None
        self.y_mean = self.y_std = 0.0
        self.n_unparsable = 0

    def _build(self) -> nn.Module:
        p = self.p
        if self.kind == "gin":
            return _GinRegressor(p["hidden_channels"], p["num_layers"], p["dropout"])
        if self.kind == "attentivefp":
            return _AttentiveFPRegressor(p["hidden_channels"], p["num_layers"],
                                         p["num_timesteps"], p["dropout"])
        raise ValueError(f"unknown PyG model '{self.kind}'")

    def _dataset(self, smiles: Sequence[str], y: Sequence[float] | None):
        ys = [0.0] * len(smiles) if y is None else list(y)
        out, bad = [], 0
        for s, yy in zip(smiles, ys):
            d = smiles_to_data(s, float(yy))
            if d is None:
                bad += 1
            else:
                out.append(d)
        self.n_unparsable = bad
        return out

    def fit(self, smiles: Sequence[str], y: Sequence[float]) -> "PyGRegressor":
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        y = np.asarray(y, dtype=float)
        
        self.y_mean, self.y_std = float(y.mean()), float(y.std() or 1.0)
        ds = self._dataset(smiles, (y - self.y_mean) / self.y_std)
        if not ds:
            raise RuntimeError("no valid molecules in training data")
        g = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(ds, batch_size=int(self.p["batch_size"]),
                            shuffle=True, generator=g, drop_last=False)
        self.model = self._build().to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=float(self.p["lr"]),
                               weight_decay=float(self.p.get("weight_decay", 0.0)))
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(int(self.p["epochs"])):
            for batch in loader:
                batch = batch.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss_fn(self.model(batch), batch.y.view(-1)).backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, smiles: Sequence[str]) -> np.ndarray:
        assert self.model is not None, "fit first"
        self.model.eval()
        
        
        out = np.full(len(smiles), np.nan, dtype=float)
        keep, ds = [], []
        for i, s in enumerate(smiles):
            d = smiles_to_data(s, 0.0)
            if d is not None:
                keep.append(i); ds.append(d)
        if ds:
            preds = []
            for batch in DataLoader(ds, batch_size=256, shuffle=False):
                preds.append(self.model(batch.to(self.device)).cpu().numpy())
            out[keep] = np.concatenate(preds) * self.y_std + self.y_mean
        return out


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
class DMPNNRegressor:
    """Documentation."""

    def __init__(self, params: dict[str, Any], *, seed: int, device: str):
        self.p, self.seed, self.device = dict(params), seed, device
        self.model = None
        self.n_unparsable = 0

    def _points(self, smiles: Sequence[str], y: Sequence[float] | None):
        from chemprop import data as cpdata
        ys = [0.0] * len(smiles) if y is None else list(y)
        pts, keep, bad = [], [], 0
        for i, (s, yy) in enumerate(zip(smiles, ys)):
            if Chem.MolFromSmiles(s) is None:
                bad += 1
                continue
            pts.append(cpdata.MoleculeDatapoint.from_smi(s, np.array([float(yy)])))
            keep.append(i)
        self.n_unparsable = bad
        return pts, keep

    def fit(self, smiles: Sequence[str], y: Sequence[float]) -> "DMPNNRegressor":
        import lightning.pytorch as pl
        from chemprop import data as cpdata, models as cpmodels, nn as cpnn

        pl.seed_everything(self.seed, workers=True)
        pts, _ = self._points(smiles, y)
        if not pts:
            raise RuntimeError("no valid molecules in training data")
        ds = cpdata.MoleculeDataset(pts)
        ds.cache = True
        
        
        scaler = ds.normalize_targets()
        loader = cpdata.build_dataloader(ds, batch_size=int(self.p["batch_size"]),
                                         num_workers=0, shuffle=True, seed=self.seed)
        mp = cpnn.BondMessagePassing(d_h=int(self.p["message_hidden_dim"]),
                                     depth=int(self.p["depth"]),
                                     dropout=float(self.p["dropout"]))
        ffn = cpnn.RegressionFFN(
            input_dim=int(self.p["message_hidden_dim"]),
            hidden_dim=int(self.p["ffn_hidden_dim"]),
            n_layers=int(self.p["ffn_num_layers"]),
            dropout=float(self.p["dropout"]),
            output_transform=cpnn.UnscaleTransform.from_standard_scaler(scaler),
        )
        self.model = cpmodels.MPNN(message_passing=mp, agg=cpnn.MeanAggregation(),
                                   predictor=ffn, batch_norm=True,
                                   max_lr=float(self.p["max_lr"]))
        self.trainer = pl.Trainer(
            max_epochs=int(self.p["epochs"]), accelerator="gpu" if "cuda" in self.device else "cpu",
            devices=1, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            
            
            
            
            
            deterministic=True,
        )
        self.trainer.fit(self.model, loader)
        return self

    def predict(self, smiles: Sequence[str]) -> np.ndarray:
        from chemprop import data as cpdata
        assert self.model is not None, "fit first"
        pts, keep = self._points(smiles, None)
        out = np.full(len(smiles), np.nan, dtype=float)
        if pts:
            loader = cpdata.build_dataloader(cpdata.MoleculeDataset(pts), batch_size=256,
                                             num_workers=0, shuffle=False)
            preds = self.trainer.predict(self.model, loader)
            out[keep] = torch.cat([p for p in preds]).view(-1).cpu().numpy()
        return out


def build_gnn(name: str, params: dict[str, Any], *, seed: int,
              device: str = "cuda") -> Any:
    """Documentation."""
    if name == "dmpnn":
        return DMPNNRegressor(params, seed=seed, device=device)
    if name in ("attentivefp", "gin"):
        return PyGRegressor(name, params, seed=seed, device=device)
    raise ValueError(f"unknown GNN '{name}' supported: {GNN_NAMES}")
