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