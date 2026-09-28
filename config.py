from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    ct_dim: int = 768
    wsi_dim: int = 768
    node_dim: int = 256
    topk_per_ct: int = 10
    hg_layers: int = 1
    dropout: float = 0.1
    wsi_search_chunk: int = 2048

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1337
    train_ratio: float = 5 / 6
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    cox_batch_size: int = 16
    require_event_in_batch: bool = True
    min_epoch_for_checkpoint: int = 1
    grad_clip_norm: float = 5.0
