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