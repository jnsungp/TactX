import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from simclr.modules import NT_Xent
except Exception:
    NT_Xent = None


def loss_vector_embedding(z_i, z_j):
    loss = torch.nn.functional.mse_loss(z_i, z_j)
    return loss


def kl_divergence_standard(mu, logvar):
    """KL[ q(z|x) || N(0, I) ] averaged over batch and latent dims.

    mu, logvar: (B, D)
    Returns: scalar tensor.
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


class CrossModalVAELoss(nn.Module):
    """Combined loss for CrossModalVAE: InfoNCE(mu) + lambda_recon * L_recon + beta * KL.

    - InfoNCE is applied on the *means* of the VAE posteriors (mu_i, mu_j), not
      the reparameterized samples. This avoids injecting sampling noise into the
      contrastive alignment signal (tactile data is already noisy).
    - 4-branch L1 reconstruction decoded from the variational latent z (NOT h),
      forcing all reconstruction-relevant information through the shared bottleneck:
          recon_ii (D_img(z_i) vs x_i),   recon_ji (D_img(z_j) vs x_i),
          recon_jj (D_mag(z_j) vs x_j),   recon_ij (D_mag(z_i) vs x_j).
      The cross branches pull z_i and z_j into the same space (shared code).
    - KL regularizes both posteriors toward N(0, I).
    - beta is set externally per-epoch (warm-up schedule).

    forward(out, x_i, x_j, beta) returns dict with sub-losses and 'total'.
    """

    def __init__(self, batch_size, temperature, lambda_recon=1.0):
        super().__init__()
        if NT_Xent is None:
            raise ImportError("simclr.modules.NT_Xent is required for CrossModalVAELoss")
        self.nt_xent = NT_Xent(batch_size, temperature, 1)
        self.lambda_recon = float(lambda_recon)

    def forward(self, out, x_i, x_j, beta=0.0):
        """
        Args:
            out: dict produced by CrossModalVAE.forward (must contain mu_i, logvar_i,
                 mu_j, logvar_j, recon_ii, recon_ji, recon_jj, recon_ij).
            x_i: Daimon image batch (B, C, H, W).
            x_j: eFlesh magnetic batch (B, 15).
            beta: float, KL weight applied this step (e.g. from warm-up scheduler).

        Returns a dict with scalar tensors: total, infonce, recon, kl,
        recon_ii, recon_ji, recon_jj, recon_ij, kl_i, kl_j, beta.
        """
        mu_i, mu_j = out["mu_i"], out["mu_j"]
        logvar_i, logvar_j = out["logvar_i"], out["logvar_j"]

        # InfoNCE on normalized means (NT_Xent internally uses cosine similarity,
        # so normalization is implicit, but we pass mu directly).
        L_info = self.nt_xent(mu_i, mu_j)

        # 4-branch L1 reconstruction (self + cross).
        r_ii = F.l1_loss(out["recon_ii"], x_i)
        r_ji = F.l1_loss(out["recon_ji"], x_i)
        r_jj = F.l1_loss(out["recon_jj"], x_j)
        r_ij = F.l1_loss(out["recon_ij"], x_j)
        L_recon = r_ii + r_ji + r_jj + r_ij

        # KL regularization.
        kl_i = kl_divergence_standard(mu_i, logvar_i)
        kl_j = kl_divergence_standard(mu_j, logvar_j)
        L_kl = kl_i + kl_j

        total = L_info + self.lambda_recon * L_recon + beta * L_kl

        return {
            "total": total,
            "infonce": L_info.detach(),
            "recon": L_recon.detach(),
            "recon_ii": r_ii.detach(), "recon_ji": r_ji.detach(),
            "recon_jj": r_jj.detach(), "recon_ij": r_ij.detach(),
            "kl": L_kl.detach(), "kl_i": kl_i.detach(), "kl_j": kl_j.detach(),
            "beta": torch.tensor(float(beta)),
        }

class CrossModalVAE3Loss(nn.Module):
    """Tri-modal VAE loss for CrossModalVAE3 (joint pair-only training).

    Forward inputs:
        outs: dict with keys 'de', 'ef', 'fd', each value is the per-pair dict
              from CrossModalVAE3.forward_pair (contains mu_a, logvar_a, z_a,
              mu_b, logvar_b, z_b, recon_aa, recon_ba, recon_bb, recon_ab,
              sensor_a, sensor_b).
        targets: dict with keys 'de', 'ef', 'fd', each value is (x_a, x_b)
                 raw input tensors used as reconstruction targets.
        beta: float, KL weight applied this step (warm-up scheduler externally).

    Loss = sum_{p in {de,ef,fd}} L_InfoNCE(mu_a^p, mu_b^p)
         + lambda_recon * sum_{p} (L1(recon_aa, x_a) + L1(recon_ba, x_a)
                                   + L1(recon_bb, x_b) + L1(recon_ab, x_b))
         + beta * sum_{p} (KL(mu_a^p, logvar_a^p) + KL(mu_b^p, logvar_b^p))

    The same N(0, I) prior anchors all three modalities. Pair-wise InfoNCE
    directly aligns DE / EF / FD; the unobserved DF alignment emerges through
    the shared prior + shared eFlesh and Daimon encoders across pairs.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size, temperature, lambda_recon=1.0):
        super().__init__()
        if NT_Xent is None:
            raise ImportError("simclr.modules.NT_Xent is required for CrossModalVAE3Loss")
        # Three independent NT_Xent objects (state-free, but keep symmetric API).
        self.nt_xent_de = NT_Xent(batch_size, temperature, 1)
        self.nt_xent_ef = NT_Xent(batch_size, temperature, 1)
        self.nt_xent_fd = NT_Xent(batch_size, temperature, 1)
        self.lambda_recon = float(lambda_recon)

    def _nt_for(self, pair):
        return {"de": self.nt_xent_de, "ef": self.nt_xent_ef,
                "fd": self.nt_xent_fd}[pair]

    @staticmethod
    def _shape_match(recon, target):
        """Reshape recon to match target shape if a singleton channel needs squeezing."""
        if recon.shape == target.shape:
            return recon
        if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
            return recon.squeeze(1)
        if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
            return recon.unsqueeze(1)
        return recon

    def forward(self, outs, targets, beta=0.0, alpha=1.0, meta=None):
        """
        Args:
            alpha: InfoNCE weight (0→1 curriculum). When 0 the model trains
                   reconstruction-only so the latent space captures contact
                   information before contrastive alignment is introduced.
            meta: accepted and ignored — kept so all tri losses share one
                  call signature (CWCL uses it).
        """
        info_per_pair = {}
        recon_per_pair = {}
        kl_per_sensor_pair = {}  # keyed by f"{pair}_{a|b}"

        L_info_total = 0.0
        L_recon_total = 0.0
        L_kl_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            x_a, x_b = targets[pair]

            mu_a, mu_b = out["mu_a"], out["mu_b"]
            logvar_a, logvar_b = out["logvar_a"], out["logvar_b"]

            L_info = self._nt_for(pair)(mu_a, mu_b)
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()

            r_aa = F.l1_loss(self._shape_match(out["recon_aa"], x_a), x_a)
            r_ba = F.l1_loss(self._shape_match(out["recon_ba"], x_a), x_a)
            r_bb = F.l1_loss(self._shape_match(out["recon_bb"], x_b), x_b)
            r_ab = F.l1_loss(self._shape_match(out["recon_ab"], x_b), x_b)
            L_recon_pair = r_aa + r_ba + r_bb + r_ab
            L_recon_total = L_recon_total + L_recon_pair
            recon_per_pair[pair] = {
                "total": L_recon_pair.detach(),
                "recon_aa": r_aa.detach(), "recon_ba": r_ba.detach(),
                "recon_bb": r_bb.detach(), "recon_ab": r_ab.detach(),
            }

            kl_a = kl_divergence_standard(mu_a, logvar_a)
            kl_b = kl_divergence_standard(mu_b, logvar_b)
            L_kl_total = L_kl_total + kl_a + kl_b
            kl_per_sensor_pair[f"{pair}_a"] = kl_a.detach()
            kl_per_sensor_pair[f"{pair}_b"] = kl_b.detach()

        total = alpha * L_info_total + self.lambda_recon * L_recon_total + beta * L_kl_total

        info = {
            "total": total,
            "infonce": L_info_total.detach(),
            "recon": L_recon_total.detach(),
            "kl": L_kl_total.detach() if torch.is_tensor(L_kl_total) else torch.tensor(float(L_kl_total)),
            "beta": torch.tensor(float(beta)),
            "alpha_infonce": torch.tensor(float(alpha)),
        }
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = recon_per_pair[pair]["total"]
            for k in ("recon_aa", "recon_ba", "recon_bb", "recon_ab"):
                info[f"{k}_{pair}"] = recon_per_pair[pair][k]
        for k, v in kl_per_sensor_pair.items():
            info[f"kl_{k}"] = v
        return info


class CrossModalVAE3UniforceLoss(nn.Module):
    """Tri-modal VAE loss for CrossModalVAE3 — 'uniforce' variant.

    Identical to CrossModalVAE3Loss in every respect EXCEPT the cross-modal
    alignment term: instead of pair-wise InfoNCE (contrastive, with negatives)
    it uses a plain L2 (MSE) pull between the matched paired means:

        L_align = sum_{p in {de,ef,fd}}  MSE(mu_a^p, mu_b^p)

    Recon (4× L1 per pair) and KL are unchanged.

        Loss = alpha * L_align
             + lambda_recon * sum_p (L1 recon, 4 branches)
             + beta        * sum_p (KL_a + KL_b)

    Output dict keys are kept identical to CrossModalVAE3Loss (``infonce``,
    ``infonce_{pair}``, ``alpha_infonce``, ...) so the existing tri training /
    validation / logging code consumes it unchanged — for this loss those
    keys carry the L2 alignment value, not an InfoNCE value.

    Note on collapse: contrastive alignment resists collapse via negatives;
    a pure L2 pull does not. Here the *only* anti-collapse force is the
    reconstruction term — different contacts must decode to different inputs,
    so the latent cannot collapse to a point. Keep ``lambda_recon`` healthy
    relative to ``alpha`` or the means will collapse toward 0.

    __init__ signature matches CrossModalVAE3Loss (batch_size / temperature
    accepted and ignored) so get_loss can construct it the same way.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size=None, temperature=None, lambda_recon=1.0):
        super().__init__()
        self.lambda_recon = float(lambda_recon)

    @staticmethod
    def _shape_match(recon, target):
        """Reshape recon to match target shape if a singleton channel needs squeezing."""
        if recon.shape == target.shape:
            return recon
        if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
            return recon.squeeze(1)
        if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
            return recon.unsqueeze(1)
        return recon

    def forward(self, outs, targets, beta=0.0, alpha=1.0, meta=None):
        """
        Args:
            alpha: alignment weight (0→1 curriculum). When 0 the model trains
                   reconstruction-only, identical to the InfoNCE variant.
            meta: accepted and ignored — kept so all tri losses share one
                  call signature (CWCL uses it).
        """
        info_per_pair = {}
        recon_per_pair = {}
        kl_per_sensor_pair = {}  # keyed by f"{pair}_{a|b}"

        L_info_total = 0.0
        L_recon_total = 0.0
        L_kl_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            x_a, x_b = targets[pair]

            mu_a, mu_b = out["mu_a"], out["mu_b"]
            logvar_a, logvar_b = out["logvar_a"], out["logvar_b"]

            # --- uniforce: L2 alignment instead of InfoNCE ---
            L_info = F.mse_loss(mu_a, mu_b)
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()

            r_aa = F.l1_loss(self._shape_match(out["recon_aa"], x_a), x_a)
            r_ba = F.l1_loss(self._shape_match(out["recon_ba"], x_a), x_a)
            r_bb = F.l1_loss(self._shape_match(out["recon_bb"], x_b), x_b)
            r_ab = F.l1_loss(self._shape_match(out["recon_ab"], x_b), x_b)
            L_recon_pair = r_aa + r_ba + r_bb + r_ab
            L_recon_total = L_recon_total + L_recon_pair
            recon_per_pair[pair] = {
                "total": L_recon_pair.detach(),
                "recon_aa": r_aa.detach(), "recon_ba": r_ba.detach(),
                "recon_bb": r_bb.detach(), "recon_ab": r_ab.detach(),
            }

            kl_a = kl_divergence_standard(mu_a, logvar_a)
            kl_b = kl_divergence_standard(mu_b, logvar_b)
            L_kl_total = L_kl_total + kl_a + kl_b
            kl_per_sensor_pair[f"{pair}_a"] = kl_a.detach()
            kl_per_sensor_pair[f"{pair}_b"] = kl_b.detach()

        total = alpha * L_info_total + self.lambda_recon * L_recon_total + beta * L_kl_total

        info = {
            "total": total,
            "infonce": L_info_total.detach(),          # carries L2 align value
            "recon": L_recon_total.detach(),
            "kl": L_kl_total.detach() if torch.is_tensor(L_kl_total) else torch.tensor(float(L_kl_total)),
            "beta": torch.tensor(float(beta)),
            "alpha_infonce": torch.tensor(float(alpha)),
        }
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = recon_per_pair[pair]["total"]
            for k in ("recon_aa", "recon_ba", "recon_bb", "recon_ab"):
                info[f"{k}_{pair}"] = recon_per_pair[pair][k]
        for k, v in kl_per_sensor_pair.items():
            info[f"kl_{k}"] = v
        return info


def loss_vector_similarity(h_i, h_j):
    loss = torch.nn.functional.cosine_similarity(h_i, h_j)
    return loss


class CWCL(nn.Module):
    """Continuous Weighted Contrastive Loss.

    Generalizes NT-Xent to soft labels: instead of treating only the diagonal
    as positives, each off-diagonal (i, j) carries a continuous weight
    w_ij in [0, 1] reflecting how similar samples i and j are.

    Weight definition used here (episode + frame distance based):
        - same tool AND same episode: w_ij = exp(-|f_i - f_j| / sigma)
        - otherwise: 0
        - diagonal forced to 1
    (If same_episode_only=False, the episode mask is dropped and the decay
    applies within the same tool.)

    Loss form (symmetric, NT-Xent style):
        L_i->j = -sum_k softmax_w(ik) * log softmax(sim(z_i, z_j) / tau)_ik
    and averaged with the j->i direction.
    """

    def __init__(self, temperature=0.5, sigma=5.0, same_episode_only=True):
        super().__init__()
        self.temperature = float(temperature)
        self.sigma = float(sigma)
        self.same_episode_only = bool(same_episode_only)

    @torch.no_grad()
    def compute_weights(self, tool_ids, episode_ids, frame_ids):
        """Build (B, B) weight matrix from per-sample metadata.

        Args:
            tool_ids: (B,) long
            episode_ids: (B,) long
            frame_ids: (B,) long
        Returns:
            W: (B, B) float tensor on the same device, values in [0, 1].
        """
        tool_ids = tool_ids.view(-1)
        episode_ids = episode_ids.view(-1)
        frame_ids = frame_ids.view(-1).float()

        same_tool = tool_ids.unsqueeze(0) == tool_ids.unsqueeze(1)
        mask = same_tool
        if self.same_episode_only:
            same_ep = episode_ids.unsqueeze(0) == episode_ids.unsqueeze(1)
            mask = mask & same_ep

        dfr = (frame_ids.unsqueeze(0) - frame_ids.unsqueeze(1)).abs()
        W = torch.exp(-dfr / max(self.sigma, 1e-6))
        W = W * mask.float()
        # Ensure self-pair is always positive with weight 1.
        B = W.shape[0]
        eye = torch.eye(B, dtype=torch.bool, device=W.device)
        W = W.masked_fill(eye, 1.0)
        return W

    @staticmethod
    def _soft_ce(logits, W):
        """Row-wise soft cross-entropy.

        logits: (B, B) similarity logits (scaled by 1/tau)
        W: (B, B) non-negative row weights (already self-included)
        """
        log_prob = F.log_softmax(logits, dim=1)
        row_sum = W.sum(dim=1, keepdim=True).clamp_min(1e-8)
        W_norm = W / row_sum
        per_row = -(W_norm * log_prob).sum(dim=1)
        return per_row.mean()

    def forward(self, z_i, z_j, meta):
        """
        Args:
            z_i: (B, D) projections from modality A (e.g., Daimon image).
            z_j: (B, D) projections from modality B (e.g., eFlesh magnetic).
            meta: tuple of (tool_ids, episode_ids, frame_ids), each (B,) long.
        Returns:
            scalar loss.
        """
        tool_ids, episode_ids, frame_ids = meta
        z_i_n = F.normalize(z_i, dim=1)
        z_j_n = F.normalize(z_j, dim=1)

        logits_ij = (z_i_n @ z_j_n.t()) / self.temperature  # anchor i -> j
        logits_ji = (z_j_n @ z_i_n.t()) / self.temperature  # anchor j -> i

        W = self.compute_weights(tool_ids, episode_ids, frame_ids)

        loss_ij = self._soft_ce(logits_ij, W)
        loss_ji = self._soft_ce(logits_ji, W.t())
        return 0.5 * (loss_ij + loss_ji)


@torch.no_grad()
def cwcl_weight_stats(tool_ids, episode_ids, frame_ids, sigma=5.0, same_episode_only=True):
    """Diagnostic: report W mass / effective positives per row.

    Returns dict with floats.
    """
    B = tool_ids.shape[0]
    tool_ids = tool_ids.view(-1)
    episode_ids = episode_ids.view(-1)
    frame_ids = frame_ids.view(-1).float()

    same_tool = tool_ids.unsqueeze(0) == tool_ids.unsqueeze(1)
    mask = same_tool
    if same_episode_only:
        same_ep = episode_ids.unsqueeze(0) == episode_ids.unsqueeze(1)
        mask = mask & same_ep

    dfr = (frame_ids.unsqueeze(0) - frame_ids.unsqueeze(1)).abs()
    W = torch.exp(-dfr / max(float(sigma), 1e-6)) * mask.float()
    eye = torch.eye(B, dtype=torch.bool, device=W.device)
    W = W.masked_fill(eye, 1.0)

    row_mass = W.sum(dim=1)                  # includes the self weight (=1)
    off_mass = row_mass - 1.0                # soft positives beyond self
    # Effective number of positives per row ~ (sum w)^2 / sum w^2
    eff_pos = (row_mass.pow(2) / W.pow(2).sum(dim=1).clamp_min(1e-8))
    return {
        'cwcl_row_mass_mean': row_mass.mean().item(),
        'cwcl_offdiag_mass_mean': off_mass.mean().item(),
        'cwcl_effective_pos_mean': eff_pos.mean().item(),
    }


class CrossModalVAE3CWCLLoss(nn.Module):
    """Tri-modal VAE loss for CrossModalVAE3 — 'cwcl' variant.

    Identical to CrossModalVAE3Loss EXCEPT the cross-modal alignment term:
    instead of pair-wise InfoNCE (NT-Xent, only the diagonal is positive) it
    uses pair-wise CWCL — soft positives for nearby same-(object,episode)
    frames, weighted by frame distance (see the CWCL class above).

        L_align = sum_{p in {de,ef,fd}}  CWCL(mu_a^p, mu_b^p, meta^p)

    Recon (4× L1 per pair) and KL are unchanged.

        Loss = alpha * L_align
             + lambda_recon * sum_p (L1 recon, 4 branches)
             + beta        * sum_p (KL_a + KL_b)

    CWCL ⊇ InfoNCE: with sigma→0 the soft-weight matrix collapses to identity
    and this reduces exactly to CrossModalVAE3Loss. So CWCL *replaces* the
    contrastive objective — do not also add InfoNCE.

    Requires `meta`: a dict {pair: (object_ids, episode_ids, frame_ids)}, each
    a (B,) long tensor for that pair's batch. Pass an episode-aware batch
    sampler (EpisodeChunkBatchSampler) — otherwise batches rarely contain
    same-episode frames and CWCL silently degrades to InfoNCE. The returned
    info dict carries `cwcl_offdiag_mass_*` / `cwcl_effective_pos_*` so you
    can verify the soft positives are actually firing.

    Output dict keeps the CrossModalVAE3Loss keys (`infonce`, `infonce_{pair}`,
    `alpha_infonce`, ...) so the tri training / val / logging code is unchanged
    — those keys carry the CWCL value.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size=None, temperature=0.5, lambda_recon=1.0,
                 cwcl_sigma=5.0, cwcl_same_episode_only=True):
        super().__init__()
        self.lambda_recon = float(lambda_recon)
        self.cwcl_sigma = float(cwcl_sigma)
        self.cwcl_same_episode_only = bool(cwcl_same_episode_only)
        # CWCL is stateless — one instance serves all three pairs.
        self.cwcl = CWCL(temperature=temperature, sigma=cwcl_sigma,
                         same_episode_only=cwcl_same_episode_only)

    @staticmethod
    def _shape_match(recon, target):
        if recon.shape == target.shape:
            return recon
        if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
            return recon.squeeze(1)
        if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
            return recon.unsqueeze(1)
        return recon

    def forward(self, outs, targets, beta=0.0, alpha=1.0, meta=None):
        """
        Args:
            meta: dict {pair: (object_ids, episode_ids, frame_ids)}. Required.
            alpha: alignment weight (0->1 curriculum). alpha=0 -> recon-only.
        """
        if meta is None:
            raise ValueError(
                "CrossModalVAE3CWCLLoss requires `meta` "
                "{pair: (object_ids, episode_ids, frame_ids)}; got None."
            )

        info_per_pair = {}
        recon_per_pair = {}
        kl_per_sensor_pair = {}
        cwcl_stats = {}

        L_info_total = 0.0
        L_recon_total = 0.0
        L_kl_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            x_a, x_b = targets[pair]
            mu_a, mu_b = out["mu_a"], out["mu_b"]
            logvar_a, logvar_b = out["logvar_a"], out["logvar_b"]

            if pair not in meta:
                raise ValueError(f"meta is missing pair '{pair}'")
            obj_ids, ep_ids, fr_ids = meta[pair]

            # --- cwcl: soft-weighted contrastive alignment ---
            L_info = self.cwcl(mu_a, mu_b, (obj_ids, ep_ids, fr_ids))
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()
            # diagnostics: how much soft-positive mass the batch actually had.
            cwcl_stats[pair] = cwcl_weight_stats(
                obj_ids, ep_ids, fr_ids,
                sigma=self.cwcl_sigma,
                same_episode_only=self.cwcl_same_episode_only,
            )

            r_aa = F.l1_loss(self._shape_match(out["recon_aa"], x_a), x_a)
            r_ba = F.l1_loss(self._shape_match(out["recon_ba"], x_a), x_a)
            r_bb = F.l1_loss(self._shape_match(out["recon_bb"], x_b), x_b)
            r_ab = F.l1_loss(self._shape_match(out["recon_ab"], x_b), x_b)
            L_recon_pair = r_aa + r_ba + r_bb + r_ab
            L_recon_total = L_recon_total + L_recon_pair
            recon_per_pair[pair] = {
                "total": L_recon_pair.detach(),
                "recon_aa": r_aa.detach(), "recon_ba": r_ba.detach(),
                "recon_bb": r_bb.detach(), "recon_ab": r_ab.detach(),
            }

            kl_a = kl_divergence_standard(mu_a, logvar_a)
            kl_b = kl_divergence_standard(mu_b, logvar_b)
            L_kl_total = L_kl_total + kl_a + kl_b
            kl_per_sensor_pair[f"{pair}_a"] = kl_a.detach()
            kl_per_sensor_pair[f"{pair}_b"] = kl_b.detach()

        total = alpha * L_info_total + self.lambda_recon * L_recon_total + beta * L_kl_total

        info = {
            "total": total,
            "infonce": L_info_total.detach(),          # carries CWCL value
            "recon": L_recon_total.detach(),
            "kl": L_kl_total.detach() if torch.is_tensor(L_kl_total) else torch.tensor(float(L_kl_total)),
            "beta": torch.tensor(float(beta)),
            "alpha_infonce": torch.tensor(float(alpha)),
        }
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = recon_per_pair[pair]["total"]
            for k in ("recon_aa", "recon_ba", "recon_bb", "recon_ab"):
                info[f"{k}_{pair}"] = recon_per_pair[pair][k]
            for sk, sv in cwcl_stats[pair].items():
                info[f"{sk}_{pair}"] = torch.tensor(float(sv))
        for k, v in kl_per_sensor_pair.items():
            info[f"kl_{k}"] = v
        return info


# ---------------------------------------------------------------------------
# Intra-modal ±1 neighbor soft-positive contrastive loss
# ---------------------------------------------------------------------------
#
# 2B-style SupCon formulation. Concatenate the matched paired means
# [mu_a; mu_b] into a (2B, D) stack, then for each anchor i in [0, 2B):
#
#   - matched cross-sensor frame (i, (i+B) % 2B)               → weight 1.0
#   - same modality, same (object, episode), |Δframe| == 1     → weight w
#   - everything else off-diagonal                              → negative
#
# The ±1 neighbors are deliberately INTRA-modal: temporal smoothness happens
# inside each sensor's latent trajectory, while cross-modal ±1 pairs stay
# strictly in the denominator (negatives). Same `meta = (object_ids,
# episode_ids, frame_ids)` plumbing as CWCL, same EpisodeChunkBatchSampler
# requirement so adjacent same-episode frames co-occur in the batch
# (otherwise the ±1 mask is mostly empty and this reduces to plain NT-Xent).


def plain_infonce_pair(mu_a, mu_b, temperature):
    """Batch-size-agnostic NT-Xent on matched cross-sensor pairs.

    Equivalent to ``simclr.modules.NT_Xent(B, temperature, 1)(mu_a, mu_b)``
    but builds the (2B, 2B) mask per call, so it accepts ANY B — needed when
    paired with EpisodeChunkBatchSampler, which emits variable-size batches.
    """
    B = mu_a.shape[0]
    N = 2 * B
    z = torch.cat([F.normalize(mu_a, dim=1), F.normalize(mu_b, dim=1)], dim=0)  # (2B, D)
    sim = (z @ z.t()) / temperature                                            # (2B, 2B)

    device = sim.device
    idx = torch.arange(N, device=device)
    pair_idx = (idx + B) % N

    eye = torch.eye(N, dtype=torch.bool, device=device)
    neg_inf = torch.finfo(sim.dtype).min
    logits = sim.masked_fill(eye, neg_inf)                # remove self from denominator
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_prob[idx, pair_idx].mean()


def temporal_cos_smoothness_per_pair(mu_a, mu_b, obj_ids, ep_ids, fr_ids,
                                     same_episode_only=True):
    """Intra-modal ±1 cosine smoothness for a paired batch.

    For every (i, j) in the batch with i < j, same (object, episode), and
    |frame_i - frame_j| == 1, accumulate
        (1 - cos(mu_a[i], mu_a[j]))  and  (1 - cos(mu_b[i], mu_b[j])).
    Each sensor is averaged over the neighbor pairs it sees, then the two
    sensors are averaged together. Cross-modal logits are NOT touched, so
    NT-Xent gradient on the matched pair stays at full strength.

    Returns (loss, n_pairs). When no ±1 neighbor pairs exist in the batch
    the loss is a zero scalar (autograd-safe) and n_pairs == 0.
    """
    B = mu_a.shape[0]
    device = mu_a.device

    same_obj = obj_ids.unsqueeze(0) == obj_ids.unsqueeze(1)
    if same_episode_only:
        same_ep = ep_ids.unsqueeze(0) == ep_ids.unsqueeze(1)
        mask_obj_ep = same_obj & same_ep
    else:
        mask_obj_ep = same_obj
    dfr = (fr_ids.unsqueeze(0) - fr_ids.unsqueeze(1)).abs()
    # Upper triangle only — count each (i, j) pair once.
    triu = torch.triu(torch.ones(B, B, dtype=torch.bool, device=device), diagonal=1)
    pair_mask = mask_obj_ep & (dfr == 1) & triu

    n_pairs = int(pair_mask.sum().item())
    if n_pairs == 0:
        # Return a zero connected to the graph so autograd is happy.
        zero = (mu_a.sum() * 0.0) + (mu_b.sum() * 0.0)
        return zero, 0

    pi, pj = pair_mask.nonzero(as_tuple=True)
    a_i = F.normalize(mu_a[pi], dim=1)
    a_j = F.normalize(mu_a[pj], dim=1)
    b_i = F.normalize(mu_b[pi], dim=1)
    b_j = F.normalize(mu_b[pj], dim=1)

    cos_a = (a_i * a_j).sum(dim=1)        # (n_pairs,) in [-1, 1]
    cos_b = (b_i * b_j).sum(dim=1)

    loss = 0.5 * ((1.0 - cos_a).mean() + (1.0 - cos_b).mean())
    return loss, n_pairs


def neighbor_intra_infonce_pair(mu_a, mu_b, obj_ids, ep_ids, fr_ids,
                                temperature, neighbor_weight=0.5,
                                same_episode_only=True):
    """Cross-sensor InfoNCE on (mu_a, mu_b) with intra-modal ±1 neighbor
    soft positives.

    SupCon-out form: for each anchor in the 2B concatenation [mu_a; mu_b],
    positives are the matched cross-sensor frame (weight 1.0) and the
    intra-modal ±1 frame on the same (object, episode) (weight
    `neighbor_weight`). Negatives = every other off-diagonal entry.

    Returns (loss, diagnostics_dict). Per-anchor weight-normalized
    soft cross-entropy averaged over the 2B anchors. Batch-size agnostic
    (mask built per call), so this works under EpisodeChunkBatchSampler
    even when chunks make the final batch smaller than nominal.
    """
    B = mu_a.shape[0]
    N = 2 * B
    device = mu_a.device

    z = torch.cat([F.normalize(mu_a, dim=1), F.normalize(mu_b, dim=1)], dim=0)  # (2B, D)
    sim = (z @ z.t()) / temperature                                            # (2B, 2B)

    idx = torch.arange(N, device=device)
    pair_idx = (idx + B) % N                          # matched cross-sensor index

    # Meta is per matched slot — replicate for both modality halves so a
    # (2B, 2B) comparison knows each row/column's (obj, ep, frame, modality).
    obj2 = torch.cat([obj_ids, obj_ids], dim=0)        # (2B,)
    ep2 = torch.cat([ep_ids, ep_ids], dim=0)
    fr2 = torch.cat([fr_ids, fr_ids], dim=0)
    mod = torch.cat([
        torch.zeros(B, dtype=torch.long, device=device),
        torch.ones(B, dtype=torch.long, device=device),
    ], dim=0)                                          # 0 = sensor_a, 1 = sensor_b

    same_obj = obj2.unsqueeze(0) == obj2.unsqueeze(1)
    same_obj_ep = same_obj
    if same_episode_only:
        same_obj_ep = same_obj_ep & (ep2.unsqueeze(0) == ep2.unsqueeze(1))
    same_mod = mod.unsqueeze(0) == mod.unsqueeze(1)
    dfr = (fr2.unsqueeze(0) - fr2.unsqueeze(1)).abs()
    intra_neighbor = same_obj_ep & same_mod & (dfr == 1)

    # Positive weight matrix
    W = torch.zeros(N, N, dtype=torch.float32, device=device)
    W[idx, pair_idx] = 1.0                             # matched cross-sensor
    W = torch.where(intra_neighbor, torch.full_like(W, float(neighbor_weight)), W)
    eye = torch.eye(N, dtype=torch.bool, device=device)
    W = W.masked_fill(eye, 0.0)                        # never include self

    # Denominator = every off-diagonal entry (SupCon-out form)
    denom_mask = ~eye
    neg_inf = torch.finfo(sim.dtype).min
    logits = sim.masked_fill(~denom_mask, neg_inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # Soft cross-entropy: row-normalize the positive weights, then -sum(W * log p)
    pos_mass = W.sum(dim=1).clamp_min(1e-8)
    W_norm = W / pos_mass.unsqueeze(1)
    per_row = -(W_norm * log_prob).sum(dim=1)
    loss = per_row.mean()

    with torch.no_grad():
        # Count strictly-neighbor positives per anchor (excludes matched pair).
        n_neighbors = intra_neighbor.float().sum(dim=1)
        eff_pos = (pos_mass.pow(2) / W.pow(2).sum(dim=1).clamp_min(1e-8))
        diag = {
            'neighbor_count_mean': n_neighbors.mean().detach(),
            'neighbor_row_mass_mean': pos_mass.mean().detach(),
            'neighbor_effective_pos_mean': eff_pos.mean().detach(),
        }
    return loss, diag


class CrossModalVAE3NeighborLoss(nn.Module):
    """Tri-modal VAE loss for CrossModalVAE3 — 'neighbor' variant.

    Cross-modal alignment uses ``neighbor_intra_infonce_pair`` (2B-SupCon-out
    form with intra-modal ±1 neighbor soft positives). The matched
    cross-sensor pair stays the unique cross-modal positive (weight 1.0);
    ±1 frame same-(object, episode) neighbors are added as INTRA-modal
    positives (weight `neighbor_weight`, default 0.5).

        Loss = alpha * sum_p neighbor_intra_infonce(mu_a^p, mu_b^p)
             + lambda_recon * sum_p (L1 recon, 4 branches)
             + beta        * sum_p (KL_a + KL_b)

    Requires `meta = {pair: (object_ids, episode_ids, frame_ids)}` and an
    episode-aware batch sampler so ±1 neighbors co-occur in the batch —
    otherwise the soft-positive mask is mostly empty and this degrades to
    plain InfoNCE. (train_tri auto-enables EpisodeChunkBatchSampler when
    loss=='neighbor'.) Batch-size agnostic — handles variable-size chunked
    batches from EpisodeChunkBatchSampler.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size=None, temperature=0.5, lambda_recon=1.0,
                 neighbor_weight=0.5, same_episode_only=True):
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_recon = float(lambda_recon)
        self.neighbor_weight = float(neighbor_weight)
        self.same_episode_only = bool(same_episode_only)

    @staticmethod
    def _shape_match(recon, target):
        if recon.shape == target.shape:
            return recon
        if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
            return recon.squeeze(1)
        if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
            return recon.unsqueeze(1)
        return recon

    def forward(self, outs, targets, beta=0.0, alpha=1.0, meta=None):
        if meta is None:
            raise ValueError(
                "CrossModalVAE3NeighborLoss requires `meta` "
                "{pair: (object_ids, episode_ids, frame_ids)}; got None."
            )

        info_per_pair = {}
        recon_per_pair = {}
        kl_per_sensor_pair = {}
        nbr_stats = {}

        L_info_total = 0.0
        L_recon_total = 0.0
        L_kl_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            x_a, x_b = targets[pair]
            mu_a, mu_b = out["mu_a"], out["mu_b"]
            logvar_a, logvar_b = out["logvar_a"], out["logvar_b"]

            if pair not in meta:
                raise ValueError(f"meta is missing pair '{pair}'")
            obj_ids, ep_ids, fr_ids = meta[pair]

            L_info, diag = neighbor_intra_infonce_pair(
                mu_a, mu_b, obj_ids, ep_ids, fr_ids,
                temperature=self.temperature,
                neighbor_weight=self.neighbor_weight,
                same_episode_only=self.same_episode_only,
            )
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()
            nbr_stats[pair] = diag

            r_aa = F.l1_loss(self._shape_match(out["recon_aa"], x_a), x_a)
            r_ba = F.l1_loss(self._shape_match(out["recon_ba"], x_a), x_a)
            r_bb = F.l1_loss(self._shape_match(out["recon_bb"], x_b), x_b)
            r_ab = F.l1_loss(self._shape_match(out["recon_ab"], x_b), x_b)
            L_recon_pair = r_aa + r_ba + r_bb + r_ab
            L_recon_total = L_recon_total + L_recon_pair
            recon_per_pair[pair] = {
                "total": L_recon_pair.detach(),
                "recon_aa": r_aa.detach(), "recon_ba": r_ba.detach(),
                "recon_bb": r_bb.detach(), "recon_ab": r_ab.detach(),
            }

            kl_a = kl_divergence_standard(mu_a, logvar_a)
            kl_b = kl_divergence_standard(mu_b, logvar_b)
            L_kl_total = L_kl_total + kl_a + kl_b
            kl_per_sensor_pair[f"{pair}_a"] = kl_a.detach()
            kl_per_sensor_pair[f"{pair}_b"] = kl_b.detach()

        total = alpha * L_info_total + self.lambda_recon * L_recon_total + beta * L_kl_total

        info = {
            "total": total,
            "infonce": L_info_total.detach(),          # carries neighbor-CL value
            "recon": L_recon_total.detach(),
            "kl": L_kl_total.detach() if torch.is_tensor(L_kl_total) else torch.tensor(float(L_kl_total)),
            "beta": torch.tensor(float(beta)),
            "alpha_infonce": torch.tensor(float(alpha)),
        }
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = recon_per_pair[pair]["total"]
            for k in ("recon_aa", "recon_ba", "recon_bb", "recon_ab"):
                info[f"{k}_{pair}"] = recon_per_pair[pair][k]
            for sk, sv in nbr_stats[pair].items():
                info[f"{sk}_{pair}"] = sv if torch.is_tensor(sv) else torch.tensor(float(sv))
        for k, v in kl_per_sensor_pair.items():
            info[f"kl_{k}"] = v
        return info


# ---------------------------------------------------------------------------
# Baseline NT-Xent + auxiliary temporal cosine alignment
# ---------------------------------------------------------------------------
#
# Sibling of ``CrossModalVAE3NeighborLoss``: keeps the cross-modal contrast
# byte-equal to the baseline (plain NT-Xent on the matched pair) and adds
# a separate cosine alignment term on intra-modal ±1 same-(object, episode)
# frames:
#
#     L_temporal_cos = (1 - cos(mu_a[t], mu_a[t±1])) + (1 - cos(mu_b[t], mu_b[t±1]))
#
# averaged over neighbor pairs that happen to be in the batch. NT-Xent's
# negative set / gradient is NOT modified (no SupCon weight normalization),
# so retrieval / cos_gap stay close to baseline; the extra term only adds
# a gentle "adjacent frames should align" pull per sensor. When λ = 0 this
# reduces exactly to CrossModalVAE3Loss.


class CrossModalVAE3TemporalCosLoss(nn.Module):
    """Tri-modal VAE loss for CrossModalVAE3 — 'temporal_cos' variant.

    Baseline CrossModalVAE3Loss (matched-pair NT-Xent + 4-branch L1 recon +
    KL) PLUS an auxiliary intra-modal cosine alignment term:

        L_tcos = (1 - cos(mu_a[t], mu_a[t±1])) + (1 - cos(mu_b[t], mu_b[t±1]))

    restricted to ±1 same-(object, episode) frame pairs in the current batch.
    The NT-Xent term is byte-equal to the baseline — cross-modal contrast is
    NOT diluted, the negative set is untouched — so this only adds a gentle
    temporal alignment regularizer per sensor.

        Loss = alpha * sum_p [ NT-Xent(mu_a^p, mu_b^p) + temporal_cos_weight * L_tcos^p ]
             + lambda_recon * sum_p (L1 recon, 4 branches)
             + beta        * sum_p (KL_a + KL_b)

    Uses ``plain_infonce_pair`` for NT-Xent (batch-size-agnostic), so this
    works under EpisodeChunkBatchSampler (variable batch size). When λ = 0
    the loss is identical to CrossModalVAE3Loss.

    Requires `meta = {pair: (object_ids, episode_ids, frame_ids)}`. An
    episode-aware batch sampler is helpful (otherwise ±1 frames rarely
    co-occur and L_tcos hardly fires); train_tri auto-enables
    EpisodeChunkBatchSampler when loss=='temporal_cos'.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size=None, temperature=0.5, lambda_recon=1.0,
                 temporal_cos_weight=0.05, same_episode_only=True):
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_recon = float(lambda_recon)
        self.temporal_cos_weight = float(temporal_cos_weight)
        self.same_episode_only = bool(same_episode_only)

    @staticmethod
    def _shape_match(recon, target):
        if recon.shape == target.shape:
            return recon
        if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
            return recon.squeeze(1)
        if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
            return recon.unsqueeze(1)
        return recon

    def forward(self, outs, targets, beta=0.0, alpha=1.0, meta=None):
        if meta is None:
            raise ValueError(
                "CrossModalVAE3TemporalCosLoss requires `meta` "
                "{pair: (object_ids, episode_ids, frame_ids)}; got None."
            )

        info_per_pair = {}
        tcos_per_pair = {}
        tcos_n_per_pair = {}
        recon_per_pair = {}
        kl_per_sensor_pair = {}

        L_info_total = 0.0
        L_tcos_total = 0.0
        L_recon_total = 0.0
        L_kl_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            x_a, x_b = targets[pair]
            mu_a, mu_b = out["mu_a"], out["mu_b"]
            logvar_a, logvar_b = out["logvar_a"], out["logvar_b"]

            if pair not in meta:
                raise ValueError(f"meta is missing pair '{pair}'")
            obj_ids, ep_ids, fr_ids = meta[pair]

            # 1) Plain NT-Xent on matched cross-sensor pair (baseline-equal).
            L_info = plain_infonce_pair(mu_a, mu_b, self.temperature)
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()

            # 2) Auxiliary intra-modal cosine alignment on ±1 same-(obj, ep).
            L_tcos, n_pairs = temporal_cos_smoothness_per_pair(
                mu_a, mu_b, obj_ids, ep_ids, fr_ids,
                same_episode_only=self.same_episode_only,
            )
            L_tcos_total = L_tcos_total + L_tcos
            tcos_per_pair[pair] = L_tcos.detach() if torch.is_tensor(L_tcos) else torch.tensor(float(L_tcos))
            tcos_n_per_pair[pair] = n_pairs

            r_aa = F.l1_loss(self._shape_match(out["recon_aa"], x_a), x_a)
            r_ba = F.l1_loss(self._shape_match(out["recon_ba"], x_a), x_a)
            r_bb = F.l1_loss(self._shape_match(out["recon_bb"], x_b), x_b)
            r_ab = F.l1_loss(self._shape_match(out["recon_ab"], x_b), x_b)
            L_recon_pair = r_aa + r_ba + r_bb + r_ab
            L_recon_total = L_recon_total + L_recon_pair
            recon_per_pair[pair] = {
                "total": L_recon_pair.detach(),
                "recon_aa": r_aa.detach(), "recon_ba": r_ba.detach(),
                "recon_bb": r_bb.detach(), "recon_ab": r_ab.detach(),
            }

            kl_a = kl_divergence_standard(mu_a, logvar_a)
            kl_b = kl_divergence_standard(mu_b, logvar_b)
            L_kl_total = L_kl_total + kl_a + kl_b
            kl_per_sensor_pair[f"{pair}_a"] = kl_a.detach()
            kl_per_sensor_pair[f"{pair}_b"] = kl_b.detach()

        L_align_total = L_info_total + self.temporal_cos_weight * L_tcos_total
        total = alpha * L_align_total + self.lambda_recon * L_recon_total + beta * L_kl_total

        info = {
            "total": total,
            "infonce": L_info_total.detach(),          # plain NT-Xent (baseline-equal)
            "recon": L_recon_total.detach(),
            "kl": L_kl_total.detach() if torch.is_tensor(L_kl_total) else torch.tensor(float(L_kl_total)),
            "beta": torch.tensor(float(beta)),
            "alpha_infonce": torch.tensor(float(alpha)),
        }
        info["temporal_cos_total"] = (L_tcos_total.detach() if torch.is_tensor(L_tcos_total)
                                      else torch.tensor(float(L_tcos_total)))
        info["temporal_cos_weight"] = torch.tensor(float(self.temporal_cos_weight))
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = recon_per_pair[pair]["total"]
            for k in ("recon_aa", "recon_ba", "recon_bb", "recon_ab"):
                info[f"{k}_{pair}"] = recon_per_pair[pair][k]
            info[f"temporal_cos_{pair}"] = tcos_per_pair[pair]
            info[f"temporal_cos_pair_count_{pair}"] = torch.tensor(float(tcos_n_per_pair[pair]))
        for k, v in kl_per_sensor_pair.items():
            info[f"kl_{k}"] = v
        return info


# ---------------------------------------------------------------------------
# Debiased (false-negative-masked) InfoNCE
# ---------------------------------------------------------------------------
#
# Plain InfoNCE/NT-Xent treats *every* off-diagonal pair in the batch as a
# negative. In this dataset that is wrong in two systematic ways:
#
#   1. Same (object, episode): every frame of one episode is the SAME physical
#      contact point pressed to slightly different depths — the contact-max
#      plateau is essentially one state — so two same-episode frames are NOT a
#      real negative. We therefore only keep different-(object, episode) frames
#      as negatives ("different-episode negative" rule).
#
#   2. No-contact ("zero") frames: each episode begins/ends with ~20 no-contact
#      frames. Two zero frames are the SAME 'no contact' state regardless of
#      which object/episode they came from, so a zero-zero pair is a false
#      negative even across episodes. We mask zero-zero pairs out of every
#      anchor's denominator.
#
# "zero" is read per sensor from the contact labels written by
# scripts/build_contact_labels.py (contact_a for the a-modality half of the
# batch, contact_b for the b-modality half), so the mask matches each
# embedding's own sensor.
#
# This is supervised false-negative cancellation (cf. Chuang et al. 2020,
# "Debiased Contrastive Learning"; Huynh et al. 2022, "Boosting Contrastive
# Self-Supervised Learning with False Negative Cancellation"; Khosla et al.
# 2020, "Supervised Contrastive Learning"): we never relabel anything as a
# *positive* (the only positive stays the matched cross-sensor frame), we only
# remove pairs we can prove are not genuine negatives. Removing false negatives
# can only relax the contrastive push; it cannot collapse the space because the
# matched-frame positive and the 4-branch reconstruction term are untouched.

_PAIR_DIR = {'de': 'de_pair', 'ef': 'ef_pair', 'fd': 'fd_pair'}


class ContactLabelLookup:
    """Per-(pair, object_id, episode, frame) no-contact flags for the
    debiased loss.

    Loads the per-episode label files written by scripts/build_contact_labels.py
    (``<root>/<pair_dir>/<object>/ep****.pt`` with keys ``frames``,
    ``contact_a``, ``contact_b``) and exposes a batch lookup keyed by the SAME
    (object_id, episode_id, frame_id) tuples the dataset emits as ``meta``.

    object_id is the index into ``object_names_per_pair[pair]``, which MUST be
    the base CrossModalPairDataset.object_names list (the dataset returns that
    exact index from __getitem__). Building the keys with the same ordering
    means no name translation is needed at batch time — this mirrors how
    scripts/eval_cross_modal_vae3_new.py matches labels to μ samples.
    """

    def __init__(self, label_root, object_names_per_pair):
        self.label_root = label_root
        # pair -> {(obj_id, ep, fr): (zero_a, zero_b)}  (zero == no contact)
        self._maps = {}
        self.n_labels = {}
        for pair, object_names in object_names_per_pair.items():
            m = self._load_pair(label_root, pair, object_names)
            self._maps[pair] = m
            self.n_labels[pair] = len(m)

    @staticmethod
    def _load_pair(label_root, pair, object_names):
        pair_dir = _PAIR_DIR[pair]
        m = {}
        for obj_id, obj in enumerate(object_names):
            ep_dir = os.path.join(label_root, pair_dir, obj)
            if not os.path.isdir(ep_dir):
                continue
            for f in sorted(glob.glob(os.path.join(ep_dir, 'ep*.pt'))):
                d = torch.load(f, map_location='cpu', weights_only=False)
                ep = int(d['episode'])
                frames = np.asarray(d['frames'])
                ca = np.asarray(d['contact_a']).astype(np.int64)
                cb = np.asarray(d['contact_b']).astype(np.int64)
                for fr, a, b in zip(frames, ca, cb):
                    # zero == NO contact == (contact == 0)
                    m[(obj_id, ep, int(fr))] = (a == 0, b == 0)
        return m

    @torch.no_grad()
    def zero_flags(self, pair, object_ids, episode_ids, frame_ids):
        """Return (zero_a, zero_b, n_found).

        zero_a / zero_b are bool tensors (B,) on the meta device. A frame whose
        label is missing defaults to False (treated as 'has contact' so it stays
        eligible as a negative — we never silently drop a negative we cannot
        verify is a false one)."""
        m = self._maps.get(pair, {})
        obj = object_ids.detach().cpu().numpy()
        ep = episode_ids.detach().cpu().numpy()
        fr = frame_ids.detach().cpu().numpy()
        B = obj.shape[0]
        za = np.zeros(B, dtype=bool)
        zb = np.zeros(B, dtype=bool)
        n_found = 0
        for i in range(B):
            v = m.get((int(obj[i]), int(ep[i]), int(fr[i])))
            if v is not None:
                za[i], zb[i] = v
                n_found += 1
        dev = object_ids.device
        return (torch.as_tensor(za, device=dev),
                torch.as_tensor(zb, device=dev),
                n_found)


def debiased_infonce_pair(mu_a, mu_b, group_ids, zero_a, zero_b, temperature,
                          mask_same_episode=True, mask_zero_zero=True):
    """Cross-sensor InfoNCE on (mu_a, mu_b) with false-negative masking.

    Args:
        mu_a, mu_b: (B, D) matched paired means; (mu_a[i], mu_b[i]) is the only
            positive for index i (same frame seen by the two sensors).
        group_ids: (B,) long, one id per (object, episode). Same id == same
            physical contact point.
        zero_a, zero_b: (B,) bool, True where that sensor reads no-contact.
        temperature: NT-Xent temperature (cosine sim is divided by it).
        mask_same_episode: drop same-group off-diagonal pairs from negatives.
        mask_zero_zero: drop pairs where both ends are no-contact.

    With both masks False this is exactly NT-Xent (2B formulation: intra-modal
    a-a / b-b and cross-modal a-b negatives all included). Returns
    (loss, diagnostics_dict). Mean log-loss over the 2B anchors.
    """
    B = mu_a.shape[0]
    N = 2 * B
    z = torch.cat([F.normalize(mu_a, dim=1), F.normalize(mu_b, dim=1)], dim=0)  # (2B, D)
    sim = (z @ z.t()) / temperature                                            # (2B, 2B)

    device = sim.device
    idx = torch.arange(N, device=device)
    pos_idx = (idx + B) % N                          # i<B -> i+B ; else -> i-B

    g = torch.cat([group_ids, group_ids], dim=0)     # (2B,)
    zero = torch.cat([zero_a, zero_b], dim=0)        # (2B,) bool

    # Candidate negatives: every off-diagonal entry except the positive.
    neg_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    neg_mask[idx, pos_idx] = False
    if mask_same_episode:
        neg_mask = neg_mask & (g.unsqueeze(0) != g.unsqueeze(1))
    both_zero = zero.unsqueeze(0) & zero.unsqueeze(1)
    if mask_zero_zero:
        neg_mask = neg_mask & (~both_zero)

    # Denominator = valid negatives + the (always-kept) positive.
    denom_mask = neg_mask.clone()
    denom_mask[idx, pos_idx] = True

    neg_inf = torch.finfo(sim.dtype).min
    logits = sim.masked_fill(~denom_mask, neg_inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -log_prob[idx, pos_idx].mean()

    with torch.no_grad():
        diag = {
            'neg_per_anchor': neg_mask.sum(dim=1).float().mean().detach(),
            'zero_frac': zero.float().mean().detach(),
        }
    return loss, diag


class CrossModalVAE3DebiasedLoss(nn.Module):
    """Tri-modal VAE loss for CrossModalVAE3 — 'debiased' variant.

    Identical to CrossModalVAE3Loss EXCEPT the cross-modal alignment term: the
    pair-wise NT-Xent is replaced by ``debiased_infonce_pair`` — InfoNCE whose
    denominator drops two kinds of provable false negatives:

        * same (object, episode) frames  (different-episode-only negatives)
        * zero-zero (no-contact / no-contact) frames, per sensor

    Recon (4× L1 per pair) and KL are unchanged:

        Loss = alpha * sum_p debiased_infonce(mu_a^p, mu_b^p)
             + lambda_recon * sum_p (L1 recon, 4 branches)
             + beta        * sum_p (KL_a + KL_b)

    Requires per-frame contact labels: call ``set_contact_lookup`` with a
    ``ContactLabelLookup`` before the first forward. ``meta`` must be a dict
    {pair: (object_ids, episode_ids, frame_ids)} (the dataset already returns
    that triple) so the loss can fetch each frame's no-contact flag.

    Output dict keeps the CrossModalVAE3Loss keys (``infonce``,
    ``infonce_{pair}``, ``alpha_infonce``, ...) so the tri training / val /
    logging code is unchanged — those keys carry the debiased-InfoNCE value.
    Per-pair masking diagnostics are emitted under ``debiased_*_{pair}``.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size=None, temperature=0.01, lambda_recon=1.0,
                 mask_same_episode=True, mask_zero_zero=True):
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_recon = float(lambda_recon)
        self.mask_same_episode = bool(mask_same_episode)
        self.mask_zero_zero = bool(mask_zero_zero)
        self.contact_lookup = None
        # group id = object_id * _GROUP_MULT + episode_id. Episodes per object
        # are < a few hundred; the multiplier just keeps the two fields disjoint.
        self._group_mult = 1_000_000

    def set_contact_lookup(self, lookup):
        self.contact_lookup = lookup

    @staticmethod
    def _shape_match(recon, target):
        if recon.shape == target.shape:
            return recon
        if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
            return recon.squeeze(1)
        if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
            return recon.unsqueeze(1)
        return recon

    def forward(self, outs, targets, beta=0.0, alpha=1.0, meta=None):
        """
        Args:
            meta: dict {pair: (object_ids, episode_ids, frame_ids)}. Required.
            alpha: alignment weight (0->1 curriculum). alpha=0 -> recon-only.
        """
        if meta is None:
            raise ValueError(
                "CrossModalVAE3DebiasedLoss requires `meta` "
                "{pair: (object_ids, episode_ids, frame_ids)}; got None."
            )
        if self.contact_lookup is None and self.mask_zero_zero:
            raise ValueError(
                "CrossModalVAE3DebiasedLoss needs a ContactLabelLookup for the "
                "zero-zero mask; call set_contact_lookup(...) first (or disable "
                "the zero mask)."
            )

        info_per_pair = {}
        recon_per_pair = {}
        kl_per_sensor_pair = {}
        dbg_per_pair = {}

        L_info_total = 0.0
        L_recon_total = 0.0
        L_kl_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            x_a, x_b = targets[pair]
            mu_a, mu_b = out["mu_a"], out["mu_b"]
            logvar_a, logvar_b = out["logvar_a"], out["logvar_b"]

            if pair not in meta:
                raise ValueError(f"meta is missing pair '{pair}'")
            obj_ids, ep_ids, fr_ids = meta[pair]
            group_ids = obj_ids.long() * self._group_mult + ep_ids.long()

            if self.contact_lookup is not None:
                zero_a, zero_b, n_found = self.contact_lookup.zero_flags(
                    pair, obj_ids, ep_ids, fr_ids)
            else:
                zero_a = torch.zeros_like(obj_ids, dtype=torch.bool)
                zero_b = torch.zeros_like(obj_ids, dtype=torch.bool)
                n_found = 0

            L_info, diag = debiased_infonce_pair(
                mu_a, mu_b, group_ids, zero_a, zero_b,
                temperature=self.temperature,
                mask_same_episode=self.mask_same_episode,
                mask_zero_zero=self.mask_zero_zero,
            )
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()
            diag['label_found_frac'] = torch.tensor(
                float(n_found) / max(1, obj_ids.shape[0]))
            dbg_per_pair[pair] = diag

            r_aa = F.l1_loss(self._shape_match(out["recon_aa"], x_a), x_a)
            r_ba = F.l1_loss(self._shape_match(out["recon_ba"], x_a), x_a)
            r_bb = F.l1_loss(self._shape_match(out["recon_bb"], x_b), x_b)
            r_ab = F.l1_loss(self._shape_match(out["recon_ab"], x_b), x_b)
            L_recon_pair = r_aa + r_ba + r_bb + r_ab
            L_recon_total = L_recon_total + L_recon_pair
            recon_per_pair[pair] = {
                "total": L_recon_pair.detach(),
                "recon_aa": r_aa.detach(), "recon_ba": r_ba.detach(),
                "recon_bb": r_bb.detach(), "recon_ab": r_ab.detach(),
            }

            kl_a = kl_divergence_standard(mu_a, logvar_a)
            kl_b = kl_divergence_standard(mu_b, logvar_b)
            L_kl_total = L_kl_total + kl_a + kl_b
            kl_per_sensor_pair[f"{pair}_a"] = kl_a.detach()
            kl_per_sensor_pair[f"{pair}_b"] = kl_b.detach()

        total = alpha * L_info_total + self.lambda_recon * L_recon_total + beta * L_kl_total

        info = {
            "total": total,
            "infonce": L_info_total.detach(),          # carries debiased value
            "recon": L_recon_total.detach(),
            "kl": L_kl_total.detach() if torch.is_tensor(L_kl_total) else torch.tensor(float(L_kl_total)),
            "beta": torch.tensor(float(beta)),
            "alpha_infonce": torch.tensor(float(alpha)),
        }
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = recon_per_pair[pair]["total"]
            for k in ("recon_aa", "recon_ba", "recon_bb", "recon_ab"):
                info[f"{k}_{pair}"] = recon_per_pair[pair][k]
            for dk, dv in dbg_per_pair[pair].items():
                info[f"debiased_{dk}_{pair}"] = dv
        for k, v in kl_per_sensor_pair.items():
            info[f"kl_{k}"] = v
        return info


def zero_supcon_infonce_pair(mu_a, mu_b, zero_a, zero_b, temperature):
    """Cross-sensor InfoNCE on (mu_a, mu_b) with a single zero-contact positive
    group (SupCon-style for the no-contact class only).

    For each anchor i in the 2B concatenation [mu_a; mu_b]:
      positives = {matched cross-sensor frame (i, i+B mod 2B)}
                ∪ {j != i : both zero[i] and zero[j] are True}
      negatives = every other off-diagonal entry.

    Concretely: zero-contact anchors are pulled toward EVERY other zero-contact
    frame in the batch (cross-sensor and intra-sensor), so the three encoders
    collapse all 'no contact' frames into one shared region of the latent.
    Contact anchors keep exactly one positive — their cross-sensor matched
    frame — so within-contact variation is preserved.

    No same-episode masking: at batch_size=64 with a large dataset the chance
    of two same-(object, episode) frames colliding in a batch is tiny.

    Args:
        mu_a, mu_b: (B, D) paired means.
        zero_a, zero_b: (B,) bool, True where the sensor reads no-contact.
        temperature: NT-Xent temperature (cosine sim is divided by it).

    Returns:
        (loss, diagnostics_dict). Per-anchor mean log-loss (SupCon-out form:
        positives are averaged inside each anchor, then anchors are averaged).
    """
    B = mu_a.shape[0]
    N = 2 * B
    z = torch.cat([F.normalize(mu_a, dim=1), F.normalize(mu_b, dim=1)], dim=0)  # (2B, D)
    sim = (z @ z.t()) / temperature                                            # (2B, 2B)

    device = sim.device
    idx = torch.arange(N, device=device)
    pair_idx = (idx + B) % N                          # i<B -> i+B ; else -> i-B

    zero = torch.cat([zero_a, zero_b], dim=0)         # (2B,) bool

    # Positives: matched cross-sensor pair  ∪  (zero, zero) anywhere.
    pos_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    pos_mask[idx, pair_idx] = True
    pos_mask = pos_mask | (zero.unsqueeze(0) & zero.unsqueeze(1))
    pos_mask[idx, idx] = False                        # never include self

    # Denominator: every off-diagonal entry (standard SupCon-out form).
    eye = torch.eye(N, dtype=torch.bool, device=device)
    denom_mask = ~eye

    neg_inf = torch.finfo(sim.dtype).min
    logits = sim.masked_fill(~denom_mask, neg_inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_count
    loss = -pos_log_prob.mean()

    with torch.no_grad():
        neg_mask = denom_mask & ~pos_mask
        diag = {
            'pos_per_anchor': pos_mask.sum(dim=1).float().mean().detach(),
            'neg_per_anchor': neg_mask.sum(dim=1).float().mean().detach(),
            'zero_frac': zero.float().mean().detach(),
        }
    return loss, diag


class CrossModalVAE3ZeroSupConLoss(nn.Module):
    """Tri-modal VAE loss for CrossModalVAE3 — 'zero_supcon' variant.

    Identical to CrossModalVAE3Loss except the cross-modal alignment term:
    the pair-wise NT-Xent is replaced by ``zero_supcon_infonce_pair`` — InfoNCE
    where every no-contact frame in the batch is positive to every other
    no-contact frame (in addition to the always-on matched cross-sensor pair).
    No episode masking, no zero-zero negative masking — just the extra
    positives. Contact anchors are unchanged.

        Loss = alpha * sum_p zero_supcon_infonce(mu_a^p, mu_b^p)
             + lambda_recon * sum_p (L1 recon, 4 branches)
             + beta        * sum_p (KL_a + KL_b)

    Requires per-frame contact labels: call ``set_contact_lookup`` with a
    ``ContactLabelLookup`` before the first forward. ``meta`` must be a dict
    {pair: (object_ids, episode_ids, frame_ids)} (the dataset already returns
    that triple) so the loss can fetch each frame's no-contact flag.

    Output keys match CrossModalVAE3Loss (``infonce``, ``infonce_{pair}``,
    ``alpha_infonce``, ...) so the training / val / logging code is unchanged
    — those keys carry the zero-supcon InfoNCE value. Per-pair diagnostics are
    emitted under ``zero_supcon_*_{pair}``.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size=None, temperature=0.01, lambda_recon=1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.lambda_recon = float(lambda_recon)
        self.contact_lookup = None

    def set_contact_lookup(self, lookup):
        self.contact_lookup = lookup

    @staticmethod
    def _shape_match(recon, target):
        if recon.shape == target.shape:
            return recon
        if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
            return recon.squeeze(1)
        if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
            return recon.unsqueeze(1)
        return recon

    def forward(self, outs, targets, beta=0.0, alpha=1.0, meta=None):
        """
        Args:
            meta: dict {pair: (object_ids, episode_ids, frame_ids)}. Required.
            alpha: alignment weight (0->1 curriculum). alpha=0 -> recon-only.
        """
        if meta is None:
            raise ValueError(
                "CrossModalVAE3ZeroSupConLoss requires `meta` "
                "{pair: (object_ids, episode_ids, frame_ids)}; got None."
            )
        if self.contact_lookup is None:
            raise ValueError(
                "CrossModalVAE3ZeroSupConLoss needs a ContactLabelLookup; "
                "call set_contact_lookup(...) first."
            )

        info_per_pair = {}
        recon_per_pair = {}
        kl_per_sensor_pair = {}
        dbg_per_pair = {}

        L_info_total = 0.0
        L_recon_total = 0.0
        L_kl_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            x_a, x_b = targets[pair]
            mu_a, mu_b = out["mu_a"], out["mu_b"]
            logvar_a, logvar_b = out["logvar_a"], out["logvar_b"]

            if pair not in meta:
                raise ValueError(f"meta is missing pair '{pair}'")
            obj_ids, ep_ids, fr_ids = meta[pair]

            zero_a, zero_b, n_found = self.contact_lookup.zero_flags(
                pair, obj_ids, ep_ids, fr_ids)

            L_info, diag = zero_supcon_infonce_pair(
                mu_a, mu_b, zero_a, zero_b, temperature=self.temperature,
            )
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()
            diag['label_found_frac'] = torch.tensor(
                float(n_found) / max(1, obj_ids.shape[0]))
            dbg_per_pair[pair] = diag

            r_aa = F.l1_loss(self._shape_match(out["recon_aa"], x_a), x_a)
            r_ba = F.l1_loss(self._shape_match(out["recon_ba"], x_a), x_a)
            r_bb = F.l1_loss(self._shape_match(out["recon_bb"], x_b), x_b)
            r_ab = F.l1_loss(self._shape_match(out["recon_ab"], x_b), x_b)
            L_recon_pair = r_aa + r_ba + r_bb + r_ab
            L_recon_total = L_recon_total + L_recon_pair
            recon_per_pair[pair] = {
                "total": L_recon_pair.detach(),
                "recon_aa": r_aa.detach(), "recon_ba": r_ba.detach(),
                "recon_bb": r_bb.detach(), "recon_ab": r_ab.detach(),
            }

            kl_a = kl_divergence_standard(mu_a, logvar_a)
            kl_b = kl_divergence_standard(mu_b, logvar_b)
            L_kl_total = L_kl_total + kl_a + kl_b
            kl_per_sensor_pair[f"{pair}_a"] = kl_a.detach()
            kl_per_sensor_pair[f"{pair}_b"] = kl_b.detach()

        total = alpha * L_info_total + self.lambda_recon * L_recon_total + beta * L_kl_total

        info = {
            "total": total,
            "infonce": L_info_total.detach(),          # carries zero-supcon value
            "recon": L_recon_total.detach(),
            "kl": L_kl_total.detach() if torch.is_tensor(L_kl_total) else torch.tensor(float(L_kl_total)),
            "beta": torch.tensor(float(beta)),
            "alpha_infonce": torch.tensor(float(alpha)),
        }
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = recon_per_pair[pair]["total"]
            for k in ("recon_aa", "recon_ba", "recon_bb", "recon_ab"):
                info[f"{k}_{pair}"] = recon_per_pair[pair][k]
            for dk, dv in dbg_per_pair[pair].items():
                info[f"zero_supcon_{dk}_{pair}"] = dv
        for k, v in kl_per_sensor_pair.items():
            info[f"kl_{k}"] = v
        return info


class CrossModalCLR3Loss(nn.Module):
    """Tri-modal *plain contrastive* loss for CrossModalCLR3.

    Pair-wise NT-Xent over the de / ef / fd projections — and nothing else.
    No reconstruction, no KL: this is the lean contrastive baseline.

        Loss = sum_{p in {de,ef,fd}}  NT-Xent(z_a^p, z_b^p)

    Contrastive alignment is applied at FULL weight from epoch 0. The ``alpha``
    (InfoNCE curriculum) and ``beta`` (KL) arguments are accepted only for
    call-signature compatibility with the VAE3 losses and are ignored — there
    is no recon term to warm up before, and no KL term.

    The returned info dict keeps the CrossModalVAE3Loss keys (``infonce``,
    ``recon``, ``kl``, ``infonce_{pair}``, ``recon_{pair}``, ``alpha_infonce``,
    ``beta``) so the existing tri training / validation / logging code consumes
    it unchanged — ``recon`` and ``kl`` (and their per-pair entries) are
    reported as a constant 0.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size, temperature, lambda_recon=None):
        super().__init__()
        if NT_Xent is None:
            raise ImportError("simclr.modules.NT_Xent is required for CrossModalCLR3Loss")
        # Three independent NT_Xent objects (state-free, but keep symmetric API).
        self.nt_xent_de = NT_Xent(batch_size, temperature, 1)
        self.nt_xent_ef = NT_Xent(batch_size, temperature, 1)
        self.nt_xent_fd = NT_Xent(batch_size, temperature, 1)

    def _nt_for(self, pair):
        return {"de": self.nt_xent_de, "ef": self.nt_xent_ef,
                "fd": self.nt_xent_fd}[pair]

    def forward(self, outs, targets=None, beta=0.0, alpha=1.0, meta=None):
        """
        Args:
            outs: dict {pair: per-pair dict from CrossModalCLR3.forward_pair}.
            targets, beta, alpha, meta: accepted and ignored (signature parity
                with the VAE3 losses — no recon, no KL, no curriculum here).
        """
        info_per_pair = {}
        L_info_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            L_info = self._nt_for(pair)(out["z_a"], out["z_b"])
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()

        total = L_info_total
        zero = torch.zeros((), device=total.device)

        info = {
            "total": total,
            "infonce": L_info_total.detach(),
            "recon": zero,
            "kl": zero,
            "beta": torch.tensor(0.0),
            "alpha_infonce": torch.tensor(1.0),
        }
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = zero
        return info