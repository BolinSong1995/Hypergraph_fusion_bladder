import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    obj = torch.load(args.checkpoint, map_location="cpu")
    state = (
        obj["model_state_dict"]
        if isinstance(obj, dict) and "model_state_dict" in obj
        else obj
    )

    print(f"Number of tensors: {len(state)}")
    for key, value in state.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        print(f"{key:40s} {shape}")


if __name__ == "__main__":
    main()
