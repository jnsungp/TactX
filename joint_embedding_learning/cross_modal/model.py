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


class FlexiTacEncoder(nn.Module):
    """
    Small CNN encoder for FlexiTac sensor (12 x 16 grid).

    Input layout:
        x: (B, 12, 16) raw, OR (B, 1, 12, 16). Internally normalized to (B, 1, 12, 16).

    Architecture (default, channels=[32, 64]):
        Conv2d(1 -> 32, k3, p1) -> BN -> ReLU
        Conv2d(32 -> 64, k3, p1) -> BN -> ReLU
        AdaptiveAvgPool2d((6, 8))
        Flatten -> Linear(64*6*8 -> output_dim)

    output_dim is set to match the other encoders (e.g., 512 for ResNet-18).
    """

    def __init__(self, in_h=12, in_w=16, channels=(32, 64), pool_h=6, pool_w=8,
                 output_dim=512):
        super().__init__()
        layers = []
        prev_c = 1
        for c in channels:
            layers.append(nn.Conv2d(prev_c, c, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            prev_c = c
        layers.append(nn.AdaptiveAvgPool2d((pool_h, pool_w)))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_c * pool_h * pool_w, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


class FlexiTacDecoder(nn.Module):
    """
    Small ConvTranspose2d decoder reconstructing a FlexiTac (12 x 16) grid
    from the VAE latent z (B, latent_dim).

    Architecture (default, channels=[64, 32]):
        Linear(z -> 64 * 3 * 4) -> reshape (B, 64, 3, 4)
        ConvT(64 -> 32, k4 s2 p1) -> BN -> ReLU              (3,4) -> (6,8)
        ConvT(32 -> 1,  k4 s2 p1)                            (6,8) -> (12,16)

    Output shape: (B, 1, 12, 16) by default. For loss against raw (B, 12, 16),
    callers can squeeze channel dim, but the loss helper handles both.
    """

    def __init__(self, in_dim, channels=(64, 32), out_h=12, out_w=16,
                 out_channels=1):
        super().__init__()
        assert out_h % (2 ** len(channels)) == 0 and out_w % (2 ** len(channels)) == 0, (
            f"out_h={out_h}, out_w={out_w} must be divisible by 2**{len(channels)}"
        )
        self.start_h = out_h // (2 ** len(channels))
        self.start_w = out_w // (2 ** len(channels))
        self.start_c = channels[0]
        self.fc = nn.Linear(in_dim, channels[0] * self.start_h * self.start_w)

        layers = []
        for i in range(len(channels)):
            in_c = channels[i]
            if i == len(channels) - 1:
                layers.append(nn.ConvTranspose2d(in_c, out_channels, kernel_size=4,
                                                 stride=2, padding=1))
            else:
                out_c = channels[i + 1]
                layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4,
                                                 stride=2, padding=1))
                layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        B = z.shape[0]
        x = self.fc(z).view(B, self.start_c, self.start_h, self.start_w)
        return self.deconv(x)  # (B, out_channels, out_h, out_w)


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


class CrossModalVAE3(nn.Module):
    """
    Tri-modal variational cross-modal model (Daimon + eFlesh + FlexiTac) trained
    on pair-only data: ``de_pair`` (Daimon, eFlesh), ``ef_pair`` (eFlesh,
    FlexiTac), ``fd_pair`` (FlexiTac, Daimon).

    Three modalities share a single latent space regularized to the same
    standard-normal prior ``p(z) = N(0, I)``. Each pair sees only 2 of 3
    modalities at a time, but the shared encoders, shared projection-head form,
    and shared prior pull the three marginal posteriors into the same code,
    giving transitive alignment of the third unseen pair (e.g. D vs F via E).

    Encoders:
        Daimon   x_D (B, 3, 224, 224) -> ResNet           -> h_D (B, n_features)
        eFlesh   x_E (B, 15)          -> MagneticEncoder  -> h_E (B, n_features)
        FlexiTac x_F (B, 12, 16)      -> FlexiTacEncoder  -> h_F (B, n_features)

    Projection heads (one per modality, output 2 * projection_dim for
    (mu, logvar)):
        proj_D, proj_E, proj_F

    Decoders (consume z, NOT h):
        Daimon   D_img: z -> (B, 3, 224, 224)
        eFlesh   D_mag: z -> (B, 15)
        FlexiTac D_flx: z -> (B, 1, 12, 16)

    Forward interface: ``forward_pair(x_a, x_b, name)`` where ``name`` is one of
    ``{'de', 'ef', 'fd'}``. Returns a generic dict with ``a/b``-suffixed keys so
    the loss can iterate over the three pairs uniformly:
        h_a, h_b, mu_a, logvar_a, mu_b, logvar_b, z_a, z_b,
        recon_aa, recon_ba, recon_bb, recon_ab,
        sensor_a, sensor_b
    """

    _PAIR_TABLE = {
        "de": ("D", "E"),
        "ef": ("E", "F"),
        "fd": ("F", "D"),
    }

    def __init__(self, image_encoder, magnetic_encoder, flexitac_encoder,
                 projection_dim, n_features,
                 image_out_channels=3, image_out_size=224,
                 image_decoder_channels=(256, 128, 64, 32),
                 magnetic_decoder_hidden=(256, 128, 64),
                 magnetic_out_dim=15,
                 flexitac_in_h=12, flexitac_in_w=16,
                 flexitac_decoder_channels=(64, 32),
                 flexitac_out_channels=1):
        super().__init__()
        self.n_features = n_features
        self.projection_dim = projection_dim

        self.image_encoder = image_encoder
        self.image_encoder.fc = nn.Identity()
        self.magnetic_encoder = magnetic_encoder
        self.flexitac_encoder = flexitac_encoder

        # Per-modality projection heads (each produces (mu, log_var)).
        self.proj_D = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, 2 * projection_dim, bias=False),
        )
        self.proj_E = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, 2 * projection_dim, bias=False),
        )
        self.proj_F = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, 2 * projection_dim, bias=False),
        )

        # Decoders consume z (the variational bottleneck).
        self.dec_D = ImageDecoder(
            projection_dim,
            channels=tuple(image_decoder_channels),
            out_channels=image_out_channels,
            out_size=image_out_size,
        )
        self.dec_E = MagneticDecoder(
            projection_dim,
            hidden_dims=tuple(magnetic_decoder_hidden),
            out_dim=magnetic_out_dim,
        )
        self.dec_F = FlexiTacDecoder(
            projection_dim,
            channels=tuple(flexitac_decoder_channels),
            out_h=flexitac_in_h,
            out_w=flexitac_in_w,
            out_channels=flexitac_out_channels,
        )

    @staticmethod
    def reparameterize(mu, logvar, training=True):
        if training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def _encoder_for(self, sensor):
        return {
            "D": self.image_encoder,
            "E": self.magnetic_encoder,
            "F": self.flexitac_encoder,
        }[sensor]

    def _projector_for(self, sensor):
        return {"D": self.proj_D, "E": self.proj_E, "F": self.proj_F}[sensor]

    def _decoder_for(self, sensor):
        return {"D": self.dec_D, "E": self.dec_E, "F": self.dec_F}[sensor]

    def _enc(self, x, sensor):
        h = self._encoder_for(sensor)(x)
        proj = self._projector_for(sensor)(h)
        mu, logvar = proj.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar, training=self.training)
        return h, mu, logvar, z

    def forward_pair(self, x_a, x_b, name):
        """
        Run a pair (a, b) through their respective encoders, sample z, and
        produce a 4-branch reconstruction set decoded from z.

        Args:
            x_a: input tensor for sensor_a in the pair.
            x_b: input tensor for sensor_b in the pair.
            name: one of {'de', 'ef', 'fd'}.

        Returns dict with:
            h_a, h_b               : (B, n_features)
            mu_a, logvar_a, z_a    : (B, projection_dim)
            mu_b, logvar_b, z_b    : (B, projection_dim)
            recon_aa, recon_ba     : decoded by D_a from (z_a, z_b) -> ~ x_a
            recon_bb, recon_ab     : decoded by D_b from (z_b, z_a) -> ~ x_b
            sensor_a, sensor_b     : 'D' / 'E' / 'F'
            pair                   : name
        """
        if name not in self._PAIR_TABLE:
            raise ValueError(
                f"Unknown pair '{name}', expected one of {list(self._PAIR_TABLE)}"
            )
        sa, sb = self._PAIR_TABLE[name]

        h_a, mu_a, logvar_a, z_a = self._enc(x_a, sa)
        h_b, mu_b, logvar_b, z_b = self._enc(x_b, sb)

        dec_a = self._decoder_for(sa)
        dec_b = self._decoder_for(sb)

        recon_aa = dec_a(z_a)  # self-recon (a-side)
        recon_ba = dec_a(z_b)  # cross: b's z decodes to a
        recon_bb = dec_b(z_b)  # self-recon (b-side)
        recon_ab = dec_b(z_a)  # cross: a's z decodes to b

        return {
            "pair": name, "sensor_a": sa, "sensor_b": sb,
            "h_a": h_a, "h_b": h_b,
            "mu_a": mu_a, "logvar_a": logvar_a, "z_a": z_a,
            "mu_b": mu_b, "logvar_b": logvar_b, "z_b": z_b,
            "recon_aa": recon_aa, "recon_ba": recon_ba,
            "recon_bb": recon_bb, "recon_ab": recon_ab,
        }

    def forward_self(self, x, sensor):
        """
        Single-modality forward used by Stage 1 dynamic self-recon pretraining.

        Args:
            x: input tensor for the given sensor.
            sensor: one of {'D', 'E', 'F'}.

        Returns dict with: h, mu, logvar, z, recon, sensor.
        """
        if sensor not in ("D", "E", "F"):
            raise ValueError(f"Unknown sensor '{sensor}', expected one of D/E/F")
        h, mu, logvar, z = self._enc(x, sensor)
        recon = self._decoder_for(sensor)(z)
        return {
            "sensor": sensor,
            "h": h, "mu": mu, "logvar": logvar, "z": z,
            "recon": recon,
        }

    def forward(self, batch):
        """
        Joint forward over all three pairs.

        Args:
            batch: dict with keys 'de', 'ef', 'fd', each mapping to (x_a, x_b).
        Returns:
            dict { 'de': out_de, 'ef': out_ef, 'fd': out_fd } where each value
            is a per-pair dict from forward_pair.
        """
        out = {}
        for name, (x_a, x_b) in batch.items():
            out[name] = self.forward_pair(x_a, x_b, name)
        return out
