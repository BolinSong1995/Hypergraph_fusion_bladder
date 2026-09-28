from __future__ import annotations

import argparse
import json

import torch

from ct_fusion.config import ModelConfig
from ct_fusion.inference import predict_patient
from ct_fusion.model import HypergraphFusionModel
from ct_fusion.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Run CT-Fusion inference for one patient."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ct", required=True, help="CT .pt file [4,768]")
    parser.add_argument("--wsi", required=True, help="WSI patch-feature CSV")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = HypergraphFusionModel(ModelConfig()).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    result = predict_patient(
        model,
        args.ct,
        args.wsi,
        device=device,
    )

    serializable = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in result.items()
    }
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
