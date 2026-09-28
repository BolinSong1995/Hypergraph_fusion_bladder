from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ct_fusion.config import ModelConfig, TrainConfig
from ct_fusion.dataset import CTWSIDataset, single_patient_collate
from ct_fusion.engine import evaluate_model, train_one_epoch
from ct_fusion.inference import export_scores
from ct_fusion.model import HypergraphFusionModel
from ct_fusion.utils import load_checkpoint, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train manuscript-aligned CT-Fusion hypergraph model."
    )
    parser.add_argument("--clinical-csv", required=True)
    parser.add_argument("--ct-feature-dir", required=True)
    parser.add_argument("--wsi-feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    set_seed(train_cfg.seed)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    clinical_csv = Path(args.clinical_csv)
    ct_feature_dir = Path(args.ct_feature_dir)
    wsi_feature_dir = Path(args.wsi_feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(clinical_csv)

    required = {"ID", "WSI", "PFS", "PFS_status"}
    missing = required - set(df_all.columns)
    if missing:
        raise ValueError(
            f"Clinical CSV missing columns: {sorted(missing)}"
        )

    df_all["ID"] = df_all["ID"].astype(str)
    df_all["WSI"] = df_all["WSI"].astype(str)

    def patient_has_features(row):
        return (
            (ct_feature_dir / f"{row['ID']}.pt").exists()
            and (wsi_feature_dir / f"{row['WSI']}.csv").exists()
        )

    df_all = df_all[
        df_all.apply(patient_has_features, axis=1)
    ].reset_index(drop=True)

    print(f"Patients with both CT and WSI features: {len(df_all)}")

    df_shuffled = df_all.sample(
        frac=1.0, random_state=train_cfg.seed
    ).reset_index(drop=True)

    n_train = int(round(len(df_shuffled) * train_cfg.train_ratio))
    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:].copy()

    print(f"Development cohort N1 = {len(df_train)}")
    print(f"Internal validation cohort N2 = {len(df_val)}")

    df_train[["ID", "WSI"]].to_csv(
        output_dir / "development_ids.csv", index=False
    )
    df_val[["ID", "WSI"]].to_csv(
        output_dir / "internal_validation_ids.csv", index=False
    )

    train_ds = CTWSIDataset(
        df_train,
        ct_feature_dir,
        wsi_feature_dir,
        model_cfg.ct_dim,
        model_cfg.wsi_dim,
    )
    val_ds = CTWSIDataset(
        df_val,
        ct_feature_dir,
        wsi_feature_dir,
        model_cfg.ct_dim,
        model_cfg.wsi_dim,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=single_patient_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=single_patient_collate,
    )

    model = HypergraphFusionModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    checkpoint_path = output_dir / "ct_fusion_hypergraph_best.pt"
    history = []
    best_val_cindex = -np.inf
    best_epoch = None

    for epoch in range(1, train_cfg.epochs + 1):
        mean_loss, train_cindex = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cox_batch_size=train_cfg.cox_batch_size,
            require_event_in_batch=train_cfg.require_event_in_batch,
            grad_clip_norm=train_cfg.grad_clip_norm,
        )

        val_cindex, *_ = evaluate_model(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_cindex": train_cindex,
                "val_cindex": val_cindex,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={mean_loss:.4f} | "
            f"Train C={train_cindex:.4f} | "
            f"Val C={val_cindex:.4f}"
        )

        if (
            epoch >= train_cfg.min_epoch_for_checkpoint
            and np.isfinite(val_cindex)
            and val_cindex > best_val_cindex
        ):
            best_val_cindex = val_cindex
            best_epoch = epoch

            save_checkpoint(
                checkpoint_path,
                model,
                epoch,
                val_cindex,
                train_cfg.seed,
                model_cfg,
                train_cfg,
            )
            print(
                f"  -> saved checkpoint at epoch {epoch} "
                f"(Val C={val_cindex:.4f})"
            )

        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv", index=False
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Best epoch: {best_epoch}")
    print(f"Best internal validation C-index: {best_val_cindex:.4f}")

    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    _, train_cindex = export_scores(
        model,
        df_train,
        ct_feature_dir,
        wsi_feature_dir,
        output_dir / "development_CT_Fusion_scores.csv",
        device,
    )
    _, val_cindex = export_scores(
        model,
        df_val,
        ct_feature_dir,
        wsi_feature_dir,
        output_dir / "internal_validation_CT_Fusion_scores.csv",
        device,
    )

    print(f"Frozen model development C-index: {train_cindex:.4f}")
    print(f"Frozen model internal validation C-index: {val_cindex:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
