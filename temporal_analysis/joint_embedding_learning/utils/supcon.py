"""Supervised Contrastive Loss (SupCon) for cross-modal learning.

Positives: any two embeddings from the same episode (any modality, any timestamp).
Negatives: embeddings from different episodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConCrossModalLoss(nn.Module):
    """
    SupCon-style loss adapted for two modalities.

    For each anchor, positives are all other embeddings with the same episode label
    (across both modalities). This is the right formulation when you believe all
    frames within an episode represent the same physical grasp (potentially at
    different intensities), and you only want cross-episode discrimination.

    Args:
        temperature: softmax temperature.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j, labels):
        """
        Args:
            z_i: (B, D) Daimon projections.
            z_j: (B, D) eFlesh projections.
            labels: (B,) episode IDs (long tensor).
        Returns:
            scalar loss.
        """
        B = z_i.size(0)
        device = z_i.device
        N = 2 * B

        # Stack modalities and normalize
        z = torch.cat([z_i, z_j], dim=0)      # (2B, D)
        z = F.normalize(z, dim=1)

        # Episode labels duplicated for both modalities
        labels_full = torch.cat([labels, labels], dim=0)  # (2B,)

        # Pairwise scaled similarity
        sim = z @ z.T / self.temperature      # (2B, 2B)

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Masks
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        labels_eq = (labels_full.unsqueeze(1) == labels_full.unsqueeze(0))  # (2B, 2B)
        pos_mask = labels_eq & ~self_mask     # same episode, excluding self
        not_self = ~self_mask                 # exclude self from denominator

        # log p(pos | anchor) for every pair
        exp_sim = torch.exp(sim) * not_self.float()
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        log_prob = sim - log_denom            # (2B, 2B)

        # Mean over positives per anchor
        pos_count = pos_mask.float().sum(dim=1)
        mean_log_prob = (pos_mask.float() * log_prob).sum(dim=1) / (pos_count + 1e-8)

        # Anchors with no positives (shouldn't happen if sampler gives multi-frame episodes)
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -mean_log_prob[valid].mean()
        return loss
