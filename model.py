from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class HypergraphConv(nn.Module):
    """Single normalized hypergraph convolution layer."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        de = torch.clamp(h.sum(dim=0), min=1e-6)
        dv = torch.clamp(h.sum(dim=1), min=1e-6)

        h_de = h * (1.0 / de).unsqueeze(0)
        a = h_de @ h.t()

        dv_inv_sqrt = 1.0 / torch.sqrt(dv)
        a = (dv_inv_sqrt.unsqueeze(1) * a) * dv_inv_sqrt.unsqueeze(0)

        x = a @ x
        x = self.lin(x)
        x = self.dropout(x)
        return self.act(x)


class NodeAttentionPool(nn.Module):
    """Learnable-query attention pooling over hypergraph nodes."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor):
        scores = (x @ self.query) / math.sqrt(x.shape[-1])
        weights = torch.softmax(scores, dim=0)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=0)
        return pooled, weights


class HypergraphFusionModel(nn.Module):
    """CT-Fusion hypergraph model.

    Inputs
    ------
    CT features:
        Tensor [4, 768], one embedding for each cranio-caudal CT subvolume.
    WSI features:
        Tensor [P, 768], one embedding for each tumor-region WSI patch.

    Architecture
    ------------
    1. Project CT and WSI representations into a shared 256-D latent space.
    2. Compute non-negative cosine similarity between each CT node and all WSI nodes.
    3. Select the top-K WSI patches for each CT node.
    4. Use the union of selected WSI patches as the compact pathology node set.
    5. Construct six hyperedges: one intra-CT, one intra-WSI, and four CT-centered
       cross-modal hyperedges.
    6. Apply hypergraph convolution, attention pooling, and a Cox risk head.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        self.topk = cfg.topk_per_ct
        self.search_chunk = cfg.wsi_search_chunk

        self.ct_proj = nn.Sequential(
            nn.Linear(cfg.ct_dim, cfg.node_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.node_dim),
        )
        self.wsi_proj = nn.Sequential(
            nn.Linear(cfg.wsi_dim, cfg.node_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.node_dim),
        )

        self.hglayers = nn.ModuleList(
            [
                HypergraphConv(cfg.node_dim, cfg.node_dim, dropout=cfg.dropout)
                for _ in range(cfg.hg_layers)
            ]
        )
        self.pool = NodeAttentionPool(cfg.node_dim)
        self.risk_head = nn.Sequential(
            nn.Linear(cfg.node_dim, cfg.node_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.node_dim, 1),
        )

    def _find_topk_over_all_wsi(
        self,
        ct_nodes: torch.Tensor,
        wsi_features: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Find top-K patches for each CT node across all WSI patches.

        The similarity search is chunked only for memory efficiency; all WSI
        patches remain eligible for selection.
        """
        p = wsi_features.shape[0]
        k = min(self.topk, p)
        ct_norm = F.normalize(ct_nodes.detach(), dim=1)

        best_scores = [
            torch.empty(0, device=ct_nodes.device) for _ in range(4)
        ]
        best_indices = [
            torch.empty(0, dtype=torch.long, device=ct_nodes.device)
            for _ in range(4)
        ]

        with torch.no_grad():
            for start in range(0, p, self.search_chunk):
                end = min(start + self.search_chunk, p)
                chunk_nodes = self.wsi_proj(wsi_features[start:end])
                chunk_norm = F.normalize(chunk_nodes, dim=1)
                sim = torch.clamp(ct_norm @ chunk_norm.t(), min=0.0)

                for ci in range(4):
                    candidate_scores = torch.cat(
                        [best_scores[ci], sim[ci]], dim=0
                    )
                    candidate_indices = torch.cat(
                        [
                            best_indices[ci],
                            torch.arange(
                                start,
                                end,
                                dtype=torch.long,
                                device=ct_nodes.device,
                            ),
                        ],
                        dim=0,
                    )
                    take = min(k, candidate_scores.numel())
                    values, pos = torch.topk(
                        candidate_scores, k=take, largest=True
                    )
                    best_scores[ci] = values
                    best_indices[ci] = candidate_indices[pos]

        return best_indices

    @staticmethod
    def _build_incidence(
        n_nodes: int,
        ct_idx: torch.Tensor,
        wsi_idx: torch.Tensor,
        per_ct_compact_idx: List[torch.Tensor],
        per_ct_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        h = torch.zeros(
            (n_nodes, 6), dtype=torch.float32, device=ct_idx.device
        )

        # e0: intra-CT cohesion hyperedge
        h[ct_idx, 0] = 1.0

        # e1: intra-WSI cohesion hyperedge
        h[wsi_idx, 1] = 1.0

        # e2-e5: one CT-centered cross-modal hyperedge per CT node
        for ci in range(4):
            col = 2 + ci
            h[ct_idx[ci], col] = 1.0

            local_idx = per_ct_compact_idx[ci]
            if local_idx.numel() > 0:
                global_wsi_idx = wsi_idx[local_idx]
                h[global_wsi_idx, col] = per_ct_weights[ci]

        return h

    def forward(
        self,
        ct_features: torch.Tensor,
        wsi_features: torch.Tensor,
        return_details: bool = False,
    ):
        cfg = self.config

        if ct_features.ndim != 2 or tuple(ct_features.shape) != (4, cfg.ct_dim):
            raise ValueError(
                f"Expected CT [4,{cfg.ct_dim}], got {tuple(ct_features.shape)}"
            )
        if (
            wsi_features.ndim != 2
            or wsi_features.shape[1] != cfg.wsi_dim
            or wsi_features.shape[0] == 0
        ):
            raise ValueError(
                f"Expected WSI [P,{cfg.wsi_dim}] with P>0, "
                f"got {tuple(wsi_features.shape)}"
            )

        ct_nodes = self.ct_proj(ct_features)

        # Select top-K patches against all available WSI embeddings.
        per_ct_original_idx = self._find_topk_over_all_wsi(
            ct_nodes, wsi_features
        )

        # Compact WSI node set = union of the four top-K selections.
        union_idx = torch.unique(
            torch.cat(per_ct_original_idx, dim=0), sorted=True
        )
        selected_wsi_raw = wsi_features[union_idx]

        # Re-project selected WSI patches with gradients enabled.
        wsi_nodes = self.wsi_proj(selected_wsi_raw)

        orig_to_compact: Dict[int, int] = {
            int(original_idx): compact_idx
            for compact_idx, original_idx in enumerate(union_idx.tolist())
        }

        ct_norm = F.normalize(ct_nodes, dim=1)
        wsi_norm = F.normalize(wsi_nodes, dim=1)

        per_ct_compact_idx: List[torch.Tensor] = []
        per_ct_weights: List[torch.Tensor] = []

        for ci in range(4):
            compact_idx = torch.tensor(
                [
                    orig_to_compact[int(j)]
                    for j in per_ct_original_idx[ci].tolist()
                ],
                dtype=torch.long,
                device=ct_features.device,
            )

            sims = torch.clamp(
                ct_norm[ci].unsqueeze(0)
                @ wsi_norm[compact_idx].t(),
                min=0.0,
            ).squeeze(0)

            weights = torch.softmax(sims, dim=0)
            per_ct_compact_idx.append(compact_idx)
            per_ct_weights.append(weights)

        x = torch.cat([ct_nodes, wsi_nodes], dim=0)
        ct_idx = torch.arange(0, 4, dtype=torch.long, device=x.device)
        wsi_idx = torch.arange(
            4,
            4 + wsi_nodes.shape[0],
            dtype=torch.long,
            device=x.device,
        )

        h = self._build_incidence(
            n_nodes=x.shape[0],
            ct_idx=ct_idx,
            wsi_idx=wsi_idx,
            per_ct_compact_idx=per_ct_compact_idx,
            per_ct_weights=per_ct_weights,
        )

        for layer in self.hglayers:
            x = layer(x, h)

        patient_embedding, node_attention = self.pool(x)
        risk = self.risk_head(patient_embedding).squeeze()

        if return_details:
            return {
                "risk": risk,
                "selected_wsi_indices": union_idx,
                "node_attention": node_attention,
                "incidence": h,
            }

        return risk
