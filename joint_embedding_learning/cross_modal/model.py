import torch
import torch.nn as nn
import torch.nn.functional as F


class MagneticEncoder(nn.Module):
    """
    MLP encoder for magnetic sensor data (eFlesh).
    Maps low-dimensional magnetic readings to a high-dimensional representation space.
    
    eFlesh sensor: 5 magnets × 3 axes = 15D input
    Output dimension matches the image encoder (e.g., 512 for ResNet-18, 2048 for ResNet-50)
    """

    def __init__(self, input_dim=15, hidden_dims=[64, 128, 256], output_dim=512):
        super(MagneticEncoder, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) magnetic sensor readings
        Returns:
            (B, output_dim) representation vector
        """
        return self.encoder(x)


class CrossModalCLR(nn.Module):
    """
    Cross-Modal Contrastive Learning between image-based (Daimon) and 
    magnetic-based (eFlesh) tactile sensors.

    Unlike standard SimCLR which uses a shared encoder for both views,
    this model uses SEPARATE encoders for each modality:
      - Image encoder (CNN, e.g. ResNet) for Daimon sensor
      - MLP encoder for eFlesh magnetic sensor

    Both encoders map to the same representation dimension, then
    separate projection heads map to a shared latent space where
    contrastive loss (NT-Xent) is applied.

    Architecture:
        Daimon (3, 240, 320) → ResNet → h_i (n_features)  → Projector_i → z_i (projection_dim)
        eFlesh (15,)         → MLP    → h_j (n_features)  → Projector_j → z_j (projection_dim)
                                                                    ↓
                                                              NT-Xent Loss(z_i, z_j)
    """

    def __init__(self, image_encoder, magnetic_encoder, projection_dim, n_features):
        """
        Args:
            image_encoder: CNN encoder for image data (e.g., ResNet-18/50)
            magnetic_encoder: MLP encoder for magnetic data (MagneticEncoder)
            projection_dim: dimension of the latent space (z) for contrastive loss
            n_features: dimension of the representation space (h), must match 
                        both encoders' output dimensions
        """
        super(CrossModalCLR, self).__init__()

        self.n_features = n_features

        # Image encoder (Daimon) — replace fc with Identity
        self.image_encoder = image_encoder
        self.image_encoder.fc = nn.Identity()

        # Magnetic encoder (eFlesh) — MLP
        self.magnetic_encoder = magnetic_encoder

        # Separate projection heads for each modality
        # g_img(h_i) = W2 * ReLU(W1 * h_i)
        self.projector_image = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )

        # g_mag(h_j) = W2 * ReLU(W1 * h_j)
        self.projector_magnetic = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        """
        Args:
            x_i: Daimon image tensor (B, 3, H, W)
            x_j: eFlesh magnetic tensor (B, 15)
        Returns:
            h_i: image representation (B, n_features)
            h_j: magnetic representation (B, n_features)
            z_i: image projection (B, projection_dim)
            z_j: magnetic projection (B, projection_dim)
        """
        # Encode each modality with its own encoder
        h_i = self.image_encoder(x_i)
        h_j = self.magnetic_encoder(x_j)

        # Project to shared latent space
        z_i = self.projector_image(h_i)
        z_j = self.projector_magnetic(h_j)

        return h_i, h_j, z_i, z_j


class ImageDecoder(nn.Module):
    """
    Deconvolutional decoder reconstructing a Daimon-style image from the
    VAE latent z (B, latent_dim).

    The spatial side is recovered by a stack of ConvTranspose2d blocks,
    each doubling the spatial resolution. A linear layer first maps the
    low-dim latent to a (channels[0], start_h, start_w) feature map.

    Args:
        in_dim: input latent dimension (e.g., projection_dim=16).
        channels: list of channel widths for the transposed-conv stack
                  (e.g., [256, 128, 64, 32] for 4 upsampling stages).
        out_channels: number of output image channels (3 for RGB / repeated depth).
        out_size: target square output side (e.g., 224).
    """

    def __init__(self, in_dim, channels=(256, 128, 64, 32), out_channels=3, out_size=224):
        super().__init__()
        assert len(channels) >= 1, "ImageDecoder.channels must have at least one entry"
        num_upsample = len(channels)
        assert out_size % (2 ** num_upsample) == 0, (
            f"out_size={out_size} must be divisible by 2**{num_upsample}"
        )
        start = out_size // (2 ** num_upsample)
        self.start_c = channels[0]
        self.start_h = start
        self.start_w = start

        self.fc = nn.Linear(in_dim, channels[0] * start * start)

        layers = []
        for i in range(num_upsample):
            in_c = channels[i]
            if i == num_upsample - 1:
                layers.append(nn.ConvTranspose2d(in_c, out_channels, kernel_size=4, stride=2, padding=1))
            else:
                out_c = channels[i + 1]
                layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        """z: (B, in_dim) -> x_hat: (B, out_channels, out_size, out_size)."""
        B = z.shape[0]
        x = self.fc(z).view(B, self.start_c, self.start_h, self.start_w)
        return self.deconv(x)


class MagneticDecoder(nn.Module):
    """
    MLP decoder reconstructing a magnetic (eFlesh) vector from the
    VAE latent z (B, latent_dim).

    Args:
        in_dim: input latent dimension (e.g., projection_dim=16).
        hidden_dims: MLP hidden widths.
        out_dim: output vector dimension (typically 15 for eFlesh).
    """

    def __init__(self, in_dim, hidden_dims=(256, 128, 64), out_dim=15):
        super().__init__()
        layers = []
        prev = in_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU(inplace=True))
            prev = hd
        layers.append(nn.Linear(prev, out_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class CrossModalVAE(nn.Module):
    """
    Variational cross-modal model with shared latent bottleneck and 4-branch
    reconstruction decoded from z (NOT h).

    Design rationale:
      - Each modality has its own encoder (ResNet / MLP) producing h.
      - A latent head per modality outputs (mu, log_var); z = mu + sigma*eps
        (z == mu at eval via the reparameterize trick).
      - InfoNCE is applied on mu (stable, deterministic alignment signal).
      - KL(q(z|x) || N(0, I)) regularizes each posterior.
      - Decoders consume z (the variational bottleneck), NOT h. This forces
        all reconstruction-relevant information through the low-dim latent,
        so the latent cannot be bypassed and truly becomes a shared code.

    Four reconstruction branches (self + cross, decoded from z):
        recon_ii = D_img(z_i) -> x_i   (self-image)
        recon_ji = D_img(z_j) -> x_i   (cross: magnetic z reconstructs image)
        recon_jj = D_mag(z_j) -> x_j   (self-magnetic)
        recon_ij = D_mag(z_i) -> x_j   (cross: image z reconstructs magnetic)

    Architecture:
        Daimon x_i -> ResNet      -> h_i -> proj_i -> (mu_i, logvar_i) -> z_i
        eFlesh x_j -> MagneticEnc -> h_j -> proj_j -> (mu_j, logvar_j) -> z_j
        z_i, z_j -> ImageDecoder     -> recon_ii, recon_ji
        z_i, z_j -> MagneticDecoder  -> recon_ij, recon_jj
    """

    def __init__(self, image_encoder, magnetic_encoder, projection_dim, n_features,
                 image_out_channels=3, image_out_size=224,
                 image_decoder_channels=(256, 128, 64, 32),
                 magnetic_decoder_hidden=(256, 128, 64),
                 magnetic_out_dim=15):
        super().__init__()
        self.n_features = n_features
        self.projection_dim = projection_dim

        self.image_encoder = image_encoder
        self.image_encoder.fc = nn.Identity()

        self.magnetic_encoder = magnetic_encoder

        # Individual projection heads; each outputs 2 * projection_dim (mu, log_var).
        self.projector_image = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, 2 * projection_dim, bias=False),
        )
        self.projector_magnetic = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, 2 * projection_dim, bias=False),
        )

        # Decoders take the low-dim latent z (projection_dim) as input.
        self.image_decoder = ImageDecoder(
            projection_dim,
            channels=tuple(image_decoder_channels),
            out_channels=image_out_channels,
            out_size=image_out_size,
        )
        self.magnetic_decoder = MagneticDecoder(
            projection_dim,
            hidden_dims=tuple(magnetic_decoder_hidden),
            out_dim=magnetic_out_dim,
        )

    @staticmethod
    def reparameterize(mu, logvar, training=True):
        """Sample z = mu + sigma * eps during training; z = mu at inference."""
        if training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def _split_projection(self, proj_out):
        mu, logvar = proj_out.chunk(2, dim=-1)
        # Clamp log_var to avoid numerical blow-ups during warm-up.
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def forward(self, x_i, x_j):
        """
        Args:
            x_i: Daimon image tensor (B, 3, H, W)
            x_j: eFlesh magnetic tensor (B, 15)

        Returns dict with:
            h_i, h_j:         (B, n_features)
            mu_i, logvar_i:   (B, projection_dim)
            mu_j, logvar_j:   (B, projection_dim)
            z_i, z_j:         (B, projection_dim)    sampled via reparam (z==mu at eval)
            recon_ii, recon_ji: (B, C, H, W)         image reconstructions from z
            recon_jj, recon_ij: (B, magnetic_out_dim) magnetic reconstructions from z
        """
        h_i = self.image_encoder(x_i)
        h_j = self.magnetic_encoder(x_j)

        mu_i, logvar_i = self._split_projection(self.projector_image(h_i))
        mu_j, logvar_j = self._split_projection(self.projector_magnetic(h_j))

        z_i = self.reparameterize(mu_i, logvar_i, training=self.training)
        z_j = self.reparameterize(mu_j, logvar_j, training=self.training)

        # Decoders consume z (the variational bottleneck), forcing all
        # reconstruction-relevant information through the shared latent.
        recon_ii = self.image_decoder(z_i)      # self-image
        recon_ji = self.image_decoder(z_j)      # cross: magnetic z -> image
        recon_jj = self.magnetic_decoder(z_j)   # self-magnetic
        recon_ij = self.magnetic_decoder(z_i)   # cross: image z -> magnetic

        return {
            "h_i": h_i, "h_j": h_j,
            "mu_i": mu_i, "logvar_i": logvar_i,
            "mu_j": mu_j, "logvar_j": logvar_j,
            "z_i": z_i, "z_j": z_j,
            "recon_ii": recon_ii, "recon_ji": recon_ji,
            "recon_jj": recon_jj, "recon_ij": recon_ij,
        }
