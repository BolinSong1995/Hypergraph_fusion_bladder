from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def safe_torch_load(path: str | Path):
    return torch.load(path, map_location="cpu")


def load_ct_features(
    path: str | Path,
    expected_dim: int = 768,
) -> torch.Tensor:
    """Load one patient's CT feature tensor.

    Expected shape: [4, expected_dim].
    """
    path = Path(path)
    x = safe_torch_load(path)

    if isinstance(x, dict):
        for key in ("features", "ct_features", "ct_feats"):
            if key in x:
                x = x[key]
                break
        else:
            raise ValueError(
                f"{path.name}: dict does not contain "
                "features/ct_features/ct_feats"
            )

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    x = x.float()

    if tuple(x.shape) != (4, expected_dim):
        raise ValueError(
            f"{path.name}: expected CT [4,{expected_dim}], "
            f"got {tuple(x.shape)}"
        )
    if not torch.isfinite(x).all():
        raise ValueError(f"{path.name}: CT features contain NaN/Inf.")

    return x


def load_wsi_features(
    path: str | Path,
    expected_dim: int = 768,
) -> Tuple[list[str], torch.Tensor]:
    """Load WSI patch-level embeddings.

    Preferred CSV format:
        patch_file, feature_0, ..., feature_767
    """
    path = Path(path)
    df = pd.read_csv(path)

    if df.shape[1] == expected_dim + 1:
        patch_ids = df.iloc[:, 0].astype(str).tolist()
        feat_df = df.iloc[:, 1:]
    else:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] != expected_dim:
            raise ValueError(
                f"{path.name}: expected {expected_dim} numeric WSI feature "
                f"columns, found {numeric.shape[1]}."
            )
        patch_ids = [str(i) for i in range(len(df))]
        feat_df = numeric

    x = feat_df.to_numpy(dtype=np.float32, copy=True)

    if x.ndim != 2 or x.shape[1] != expected_dim or x.shape[0] == 0:
        raise ValueError(
            f"{path.name}: expected WSI [P,{expected_dim}] with P>0, "
            f"got {x.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError(f"{path.name}: WSI features contain NaN/Inf.")

    return patch_ids, torch.from_numpy(x)


class CTWSIDataset(Dataset):
    """Dataset for paired CT and WSI feature files."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        ct_feature_dir: str | Path,
        wsi_feature_dir: str | Path,
        ct_dim: int = 768,
        wsi_dim: int = 768,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.ct_feature_dir = Path(ct_feature_dir)
        self.wsi_feature_dir = Path(wsi_feature_dir)
        self.ct_dim = ct_dim
        self.wsi_dim = wsi_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row["ID"])
        wsi_id = str(row["WSI"])

        ct_path = self.ct_feature_dir / f"{patient_id}.pt"
        wsi_path = self.wsi_feature_dir / f"{wsi_id}.csv"

        ct = load_ct_features(ct_path, expected_dim=self.ct_dim)
        _, wsi = load_wsi_features(wsi_path, expected_dim=self.wsi_dim)

        return {
            "ct": ct,
            "wsi": wsi,
            "time": torch.tensor(float(row["PFS"]), dtype=torch.float32),
            "event": torch.tensor(float(row["PFS_status"]), dtype=torch.float32),
            "patient_id": patient_id,
            "wsi_id": wsi_id,
        }


def single_patient_collate(batch):
    """Required because patients have different numbers of WSI patches."""
    if len(batch) != 1:
        raise ValueError("Use DataLoader(batch_size=1) with this collate function.")
    return batch[0]
