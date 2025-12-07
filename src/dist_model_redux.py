import einops
import torch
import torch.nn as nn
import math


class FourierTimeEmbedding(nn.Module):
    """Fourier feature embedding for encoding raw acquisition times."""
    def __init__(self, num_freqs=64, max_freq=10.0):
        """
        Args:
            num_freqs: Number of frequency bands
            max_freq: Maximum frequency for Fourier features
        """
        super().__init__()
        self.num_freqs = num_freqs
        # Create logarithmically spaced frequencies
        freq_bands = torch.logspace(0, math.log10(max_freq), num_freqs)
        self.register_buffer('freq_bands', freq_bands)

        # Output dimension is 2 * num_freqs (sin and cos for each frequency)
        self.out_dim = 2 * num_freqs

    def forward(self, acq_dts_float):
        """
        Args:
            acq_dts_float: Tensor of shape (B, T) — raw acquisition dates (in fractional years)
        Returns:
            time_emb: Tensor of shape (B, T, 2*num_freqs)
        """
        # Add feature dimension: (B, T) -> (B, T, 1)
        t = acq_dts_float.unsqueeze(-1)  # (B, T, 1)

        # Compute scaled inputs: (B, T, 1) * (num_freqs,) -> (B, T, num_freqs)
        scaled = t * self.freq_bands.view(1, 1, -1) * 2 * math.pi

        # Compute sin and cos
        sin_emb = torch.sin(scaled)  # (B, T, num_freqs)
        cos_emb = torch.cos(scaled)  # (B, T, num_freqs)

        # Concatenate: (B, T, 2*num_freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)


class FactorizedSpatioTemporalAttention(nn.Module):
    """
    Factorized attention that applies spatial and temporal attention separately.
    Based on "Is Space-Time Attention All You Need for Video Understanding?" (TimeSformer).
    """
    def __init__(self, d_model, nhead, dropout=0.1, num_layers=4, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Temporal attention layers (attention over time dimension)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers)

        # Spatial attention layers (attention over spatial dimension)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, P, d_model)
               where B=batch, T=time, P=num_patches, d_model=feature_dim
        Returns:
            Output tensor of same shape (B, T, P, d_model)
        """
        B, T, P, D = x.shape

        # 1. Temporal Attention: attend over time for each spatial location
        # Reshape to (B*P, T, D) to process each spatial location independently
        x_temporal = einops.rearrange(x, 'b t p d -> (b p) t d')
        x_temporal = self.temporal_transformer(x_temporal)  # (B*P, T, D)
        x_temporal = einops.rearrange(x_temporal, '(b p) t d -> b t p d', b=B, p=P)

        # 2. Spatial Attention: attend over space for each time step
        # Reshape to (B*T, P, D) to process each time step independently
        x_spatial = einops.rearrange(x_temporal, 'b t p d -> (b t) p d')
        x_spatial = self.spatial_transformer(x_spatial)  # (B*T, P, D)
        x_spatial = einops.rearrange(x_spatial, '(b t) p d -> b t p d', b=B, t=T)

        return x_spatial


class SpatioTemporalTransformerRedux(nn.Module):
    """
    V3: Factorized spatial-temporal attention with Fourier time embedding.
    No temporal padding - uses fixed-length sequences.
    """
    def __init__(self, model_config: dict) -> None:
        super().__init__()

        self.d_model = model_config['d_model']
        self.nhead = model_config['nhead']
        self.num_encoder_layers = model_config['num_encoder_layers']
        self.dim_feedforward = model_config['dim_feedforward']
        self.temporal_length = model_config['temporal_length']  # Fixed temporal length (no padding)
        self.dropout = model_config['dropout']
        self.activation = model_config['activation']
        self.input_size = model_config['input_size']
        self.patch_size = model_config['patch_size']
        self.num_patches = int((self.input_size / self.patch_size) ** 2)
        self.data_dim = model_config['data_dim']

        # Learnable token for NaN values
        self.nan_token = nn.Parameter(torch.randn(2) * 0.02)

        # Patch embedding
        self.embedding = nn.Linear(self.data_dim, self.d_model)

        # Spatial positional embedding (for patches within each frame)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, self.d_model))

        # Fourier-based time encoding for raw acquisition times
        self.time_encoder = FourierTimeEmbedding(
            num_freqs=model_config.get('fourier_freqs', 64),
            max_freq=model_config.get('fourier_max_freq', 10.0)
        )

        # Project Fourier features to d_model
        self.time_projection = nn.Linear(self.time_encoder.out_dim, self.d_model)

        # Factorized spatial-temporal attention
        self.factorized_attention = FactorizedSpatioTemporalAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
            num_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward
        )

        self.mean_out = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.data_dim),
        )
        self.logvar_out = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.data_dim),
        )

    def num_parameters(self) -> float:
        return sum(p.numel() for p in self.parameters())

    def replace_nans_only(self, x):
        """Replace NaNs with learned nan_token and clamp to prevent extreme values."""
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            token = self.nan_token.view(1, 1, -1, 1, 1)  # (1,1,C,1,1) broadcasts to (B,T,C,H,W)
            # Clamp token values to prevent extreme values
            token = torch.clamp(token, min=-30.0, max=10.0)
            x = torch.where(nan_mask, token, x)
        # Additional safety: clamp the entire tensor to prevent extreme values
        x = torch.clamp(x, min=-30.0, max=1.0)
        return x

    def forward(self, img_baseline: torch.Tensor, acq_dts_float: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_baseline: (B, T, C, H, W) - no padding, fixed temporal length
            acq_dts_float: (B, T) - raw acquisition dates (in fractional years)
        Returns:
            mean, logvar: Both (B, C, H, W) - prediction for post image
        """
        B, T, C, H, W = img_baseline.shape
        device = img_baseline.device
        dtype = img_baseline.dtype

        # Replace NaNs
        x = self.replace_nans_only(img_baseline)

        # Reshape to patches: (B, T, C, H, W) -> (B, T, P, D_patch)
        x = einops.rearrange(
            x, 'b t c (h ph) (w pw) -> b t (h w) (c ph pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        # Embed patches: (B, T, P, D_patch) -> (B, T, P, d_model)
        x = self.embedding(x)

        # Add spatial positional embedding
        x = x + self.spatial_pos_embed  # (B, T, P, d_model)

        # Generate temporal embeddings from raw acquisition dates
        # Replace any NaN values in acquisition dates with 0
        acq_dts_clamped = torch.where(torch.isnan(acq_dts_float), torch.zeros_like(acq_dts_float), acq_dts_float)
        temporal_emb = self.time_encoder(acq_dts_clamped)  # (B, T, 2*num_freqs)
        temporal_emb = self.time_projection(temporal_emb)  # (B, T, d_model)

        # Add temporal embedding (broadcast across patches)
        temporal_emb = temporal_emb.unsqueeze(2)  # (B, T, 1, d_model)
        x = x + temporal_emb  # (B, T, P, d_model)

        # Apply factorized spatial-temporal attention
        output = self.factorized_attention(x)  # (B, T, P, d_model)

        # Add numerical stability checks
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN/Inf detected in transformer output, applying emergency clipping")
            output = torch.clamp(output, min=-100.0, max=100.0)
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)

        # Generate mean and logvar predictions
        mean = self.mean_out(output)  # (B, T, P, D_patch)
        logvar = self.logvar_out(output)  # (B, T, P, D_patch)

        # Clamp outputs to prevent extreme values
        mean = torch.clamp(mean, min=-30.0, max=10.0)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)

        # Reshape back to image format
        mean = einops.rearrange(
            mean, 'b t (h w) (c ph pw) -> b t c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size,
            c=C, h=(H // self.patch_size), w=(W // self.patch_size)
        )
        logvar = einops.rearrange(
            logvar, 'b t (h w) (c ph pw) -> b t c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size,
            c=C, h=(H // self.patch_size), w=(W // self.patch_size)
        )

        # Return only final timestep prediction
        return mean[:, -1, ...], logvar[:, -1, ...]
