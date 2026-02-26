"""
Clay Temporal Prediction Model

This model takes pre-computed Clay embeddings (15 pre-images) and their acquisition dates,
along with a target post-date, and predicts:
1. The Clay embedding at the post-date (for embedding-space loss)
2. The SAR image at the post-date (decoded from predicted embedding, for SAR-space loss)

Architecture:
    - Input: pre_embeddings (B, 15, 1024, 32, 32), pre_dates (B, 15), post_date (B, 1)
    - Temporal Model: Predicts embedding at post_date using factorized attention
    - SAR Decoder: Decodes predicted embedding to SAR image with mean + variance

Based on dist_model_redux.py with modifications for Clay embedding input.
"""

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierTimeEmbedding(nn.Module):
    """Fourier feature embedding for encoding raw acquisition times."""

    def __init__(self, num_freqs: int = 64, max_freq: float = 10.0):
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
        self.out_dim = 2 * num_freqs  # sin and cos for each frequency

    def forward(self, acq_dts_float: torch.Tensor) -> torch.Tensor:
        """
        Args:
            acq_dts_float: Tensor of shape (B, T) — raw acquisition dates (fractional years)
        Returns:
            time_emb: Tensor of shape (B, T, 2*num_freqs)
        """
        t = acq_dts_float.unsqueeze(-1)  # (B, T, 1)
        scaled = t * self.freq_bands.view(1, 1, -1) * 2 * math.pi  # (B, T, num_freqs)
        sin_emb = torch.sin(scaled)
        cos_emb = torch.cos(scaled)
        return torch.cat([sin_emb, cos_emb], dim=-1)  # (B, T, 2*num_freqs)


class TemporalTransformerEncoder(nn.Module):
    """
    Temporal-only transformer encoder.
    Processes each spatial location independently over time.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, P, D) — batch, time, spatial patches, features
        Returns:
            (B, T, P, D)
        """
        B, T, P, D = x.shape

        # Process each spatial location over time: (B*P, T, D)
        x = einops.rearrange(x, 'b t p d -> (b p) t d')
        x = self.transformer(x)
        x = einops.rearrange(x, '(b p) t d -> b t p d', b=B, p=P)

        return x


class CrossAttentionPredictor(nn.Module):
    """
    Cross-attention module that uses post_date as a query to predict future embedding.
    Query: post_date embedding
    Key/Value: encoded pre-image sequence
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, 1, D) — post-date query embedding
            kv: (B, T, D) — encoded pre-image sequence
        Returns:
            (B, 1, D) — predicted embedding for post-date
        """
        # Cross-attention
        attn_out, _ = self.cross_attn(query, kv, kv)
        x = self.norm(query + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class SARDecoder(nn.Module):
    """
    Decodes Clay embedding (1024, 32, 32) to SAR image (2, 256, 256).
    Outputs mean and log-variance for NLL loss.

    Architecture: Progressive upsampling with ConvTranspose2d
        32 -> 64 -> 128 -> 256 (3 upsample stages, 8x total)
    """

    def __init__(self, in_channels: int = 1024, out_channels: int = 2,
                 hidden_dims: list = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 64]

        # Build decoder layers using bilinear upsample + Conv2d to avoid
        # checkerboard artifacts that ConvTranspose2d produces at stride>1.
        layers = []
        current_channels = in_channels

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(current_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
            ])
            current_channels = hidden_dim

        self.decoder = nn.Sequential(*layers)

        # Separate heads for mean and variance
        self.mean_head = nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1024, 32, 32) — Clay embedding
        Returns:
            mean: (B, 2, 256, 256) — predicted SAR mean
            logvar: (B, 2, 256, 256) — predicted SAR log-variance
        """
        features = self.decoder(x)  # (B, 64, 256, 256)

        mean = self.mean_head(features)
        logvar = self.logvar_head(features)

        # Clamp outputs for numerical stability (SAR in dB scale)
        mean = torch.clamp(mean, min=-30.0, max=10.0)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)

        return mean, logvar


class ClayTemporalPredictor(nn.Module):
    """
    Main model that predicts future Clay embeddings and decodes to SAR.

    v2 uses residual prediction: the temporal model predicts the *change* from the
    temporal mean of the pre-embeddings, and a skip connection carries the baseline
    spatial structure at full 32×32 resolution.

    Pipeline:
        1. Compute skip = 1×1 Conv(mean(pre_embeddings over time))  [full 32×32]
        2. Pool pre_embeddings spatially: (B, T, 1024, 32, 32) -> (B, T, P, D)
        3. Add temporal (Fourier) embeddings from pre_dates
        4. Encode with temporal transformer
        5. Use cross-attention with post_date query to predict future residual
        6. Upsample residual back to (B, 1024, 32, 32)
        7. pred_embedding = residual + skip
        8. Decode to SAR (B, 2, 256, 256)

    Args:
        model_config: dict with model hyperparameters
    """

    def __init__(self, model_config: dict):
        super().__init__()

        # Core dimensions
        self.d_model = model_config.get('d_model', 512)
        self.nhead = model_config.get('nhead', 8)
        self.num_encoder_layers = model_config.get('num_encoder_layers', 4)
        self.dim_feedforward = model_config.get('dim_feedforward', 2048)
        self.dropout = model_config.get('dropout', 0.1)

        # Clay embedding dimensions
        self.clay_dim = model_config.get('clay_dim', 1024)
        self.clay_spatial = model_config.get('clay_spatial', 32)

        # Spatial pooling config (32x32 -> patch_grid x patch_grid)
        self.patch_grid = model_config.get('patch_grid', 16)  # 16x16 = 256 patches
        self.num_patches = self.patch_grid ** 2

        # Pool size to go from 32 -> patch_grid
        self.pool_size = self.clay_spatial // self.patch_grid  # 32/16 = 2

        # Fourier time encoding
        self.time_encoder = FourierTimeEmbedding(
            num_freqs=model_config.get('fourier_freqs', 64),
            max_freq=model_config.get('fourier_max_freq', 10.0),
        )

        # Project Clay features to d_model
        # After pooling: each patch has clay_dim features
        self.input_projection = nn.Linear(self.clay_dim, self.d_model)

        # Project time encoding to d_model
        self.time_projection = nn.Linear(self.time_encoder.out_dim, self.d_model)

        # Spatial positional embedding
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.num_patches, self.d_model)
        )

        # Temporal transformer encoder
        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        # Learnable query token for post-date prediction
        self.post_query_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # Cross-attention predictor
        self.cross_attention = CrossAttentionPredictor(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
        )

        # Project back to Clay dimension
        self.output_projection = nn.Linear(self.d_model, self.clay_dim)

        # Upsample from (B, clay_dim, patch_grid, patch_grid) back to (B, clay_dim, 32, 32)
        # With patch_grid=16: bilinear 2× upsample + conv refinement (16×16 → 32×32)
        upsample_factor = self.clay_spatial // self.patch_grid  # 32/16 = 2
        self.spatial_upsample = nn.Sequential(
            nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(self.clay_dim, self.clay_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Skip connection: project temporal mean of pre-embeddings (full 32×32 resolution)
        # This lets the model predict residuals rather than the full embedding from scratch
        self.skip_proj = nn.Conv2d(self.clay_dim, self.clay_dim, kernel_size=1)

        # SAR decoder
        self.sar_decoder = SARDecoder(
            in_channels=self.clay_dim,
            out_channels=2,
            hidden_dims=model_config.get('decoder_hidden_dims', [512, 256, 64]),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to ViT."""
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.post_query_token, std=0.02)
        # Initialize skip_proj close to identity so residual starts near zero
        nn.init.eye_(self.skip_proj.weight.view(self.clay_dim, self.clay_dim))
        nn.init.zeros_(self.skip_proj.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        pre_embeddings: torch.Tensor,
        pre_dates: torch.Tensor,
        post_date: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pre_embeddings: (B, T, 1024, 32, 32) — Clay embeddings of pre-images
            pre_dates: (B, T) — acquisition dates of pre-images (fractional years)
            post_date: (B, 1) — acquisition date to predict (fractional years)

        Returns:
            pred_embedding: (B, 1024, 32, 32) — predicted Clay embedding
            pred_sar_mean: (B, 2, 256, 256) — predicted SAR mean
            pred_sar_logvar: (B, 2, 256, 256) — predicted SAR log-variance
        """
        B, T, C, H, W = pre_embeddings.shape
        device = pre_embeddings.device

        # 1a. Skip connection: temporal mean at full 32×32 resolution
        skip = self.skip_proj(pre_embeddings.mean(dim=1))  # (B, 1024, 32, 32)

        # 1b. Spatial pooling: (B, T, 1024, 32, 32) -> (B, T, 1024, 16, 16)
        x = einops.rearrange(pre_embeddings, 'b t c h w -> (b t) c h w')
        x = F.avg_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = einops.rearrange(x, '(b t) c h w -> b t (h w) c', b=B, t=T)
        # x: (B, T, 256, 1024)

        # 2. Project to d_model
        x = self.input_projection(x)  # (B, T, 256, d_model)

        # 3. Add spatial positional embedding
        x = x + self.spatial_pos_embed  # (B, T, P, d_model)

        # 4. Generate temporal embeddings from pre_dates
        time_emb = self.time_encoder(pre_dates)  # (B, T, 2*num_freqs)
        time_emb = self.time_projection(time_emb)  # (B, T, d_model)
        time_emb = time_emb.unsqueeze(2)  # (B, T, 1, d_model)
        x = x + time_emb  # (B, T, P, d_model)

        # 5. Temporal encoding
        x = self.temporal_encoder(x)  # (B, T, P, d_model)

        # 6. Prepare for cross-attention: pool across spatial for query context
        # For each spatial location, use cross-attention to query the future
        # Reshape: (B, T, P, D) -> (B*P, T, D)
        x_flat = einops.rearrange(x, 'b t p d -> (b p) t d')

        # 7. Create post-date query
        post_time_emb = self.time_encoder(post_date)  # (B, 1, 2*num_freqs)
        post_time_emb = self.time_projection(post_time_emb)  # (B, 1, d_model)
        query = self.post_query_token.expand(B, -1, -1) + post_time_emb  # (B, 1, d_model)

        # Expand query for each spatial location
        query_flat = einops.repeat(query, 'b 1 d -> (b p) 1 d', p=self.num_patches)

        # 8. Cross-attention: query the encoded sequence for future prediction
        pred_flat = self.cross_attention(query_flat, x_flat)  # (B*P, 1, d_model)
        pred = einops.rearrange(pred_flat, '(b p) 1 d -> b p d', b=B, p=self.num_patches)
        # pred: (B, 256, d_model)

        # 9. Project back to Clay dimension
        pred = self.output_projection(pred)  # (B, 256, 1024)

        # 10. Reshape to spatial grid and upsample
        pred = einops.rearrange(pred, 'b (h w) c -> b c h w',
                                h=self.patch_grid, w=self.patch_grid)
        # pred: (B, 1024, 16, 16)

        residual = self.spatial_upsample(pred)  # (B, 1024, 32, 32)

        # 11. Combine residual with skip connection
        pred_embedding = residual + skip  # (B, 1024, 32, 32)

        # 12. Decode to SAR
        pred_sar_mean, pred_sar_logvar = self.sar_decoder(pred_embedding)

        return pred_embedding, pred_sar_mean, pred_sar_logvar
