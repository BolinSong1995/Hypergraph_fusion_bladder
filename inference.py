from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import (
    CTWSIDataset,
    load_ct_features,
    load_wsi_features,
    single_patient_collate,
)
from .utils import harrell_c_index


@torch.no_grad()
def predict_patient(
    model,
    ct_path,
    wsi_path,
    device,
):
    model.eval()

    ct = load_ct_features(
        ct_path, expected_dim=model.config.ct_dim
    ).to(device)
    _, wsi = load_wsi_features(
        wsi_path, expected_dim=model.config.wsi_dim
    )
    wsi = wsi.to(device)

    output = model(ct, wsi, return_details=True)

    return {
        "CT_Fusion_score": float(output["risk"].detach().cpu()),
        "n_WSI_patches": int(wsi.shape[0]),
        "n_selected_WSI_patches": int(
            output["selected_wsi_indices"].numel()
        ),
        "selected_WSI_indices": (
            output["selected_wsi_indices"].detach().cpu().numpy()
        ),
    }


@torch.no_grad()
def export_scores(
    model,
    dataframe,
    ct_feature_dir,
    wsi_feature_dir,
    output_csv,
    device,
):
    dataset = CTWSIDataset(
        dataframe,
        ct_feature_dir,
        wsi_feature_dir,
        ct_dim=model.config.ct_dim,
        wsi_dim=model.config.wsi_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=single_patient_collate,
    )

    rows = []

    model.eval()
    for sample in loader:
        ct = sample["ct"].to(device)
        wsi = sample["wsi"].to(device)

        output = model(ct, wsi, return_details=True)

        rows.append(
            {
                "ID": sample["patient_id"],
                "WSI": sample["wsi_id"],
                "PFS": float(sample["time"]),
                "PFS_status": int(sample["event"]),
                "CT_Fusion_score": float(
                    output["risk"].detach().cpu()
                ),
                "n_selected_WSI_patches": int(
                    output["selected_wsi_indices"].numel()
                ),
            }
        )

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_csv, index=False)

    cindex = harrell_c_index(
        output_df["CT_Fusion_score"].values,
        output_df["PFS"].values,
        output_df["PFS_status"].values,
    )

    return output_df, cindex
