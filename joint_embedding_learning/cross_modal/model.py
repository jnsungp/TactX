import torch
import torch.nn as nn


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
