import einops
import torch
import torch.nn as nn


class SpatioTemporalTransformerRedux(nn.Module):
    def __init__(self, model_config: dict) -> None:
        super().__init__()

        self.d_model = model_config['d_model']
        self.nhead = model_config['nhead']
        self.num_encoder_layers = model_config['num_encoder_layers']
        self.dim_feedforward = model_config['dim_feedforward']
        self.max_seq_len = model_config['max_seq_len']
        self.dropout = model_config['dropout']
        self.activation = model_config['activation']
        self.input_size = model_config['input_size']
        self.patch_size = model_config['patch_size']
        self.num_patches = int((self.input_size / self.patch_size) ** 2)
        self.data_dim = model_config['data_dim']
        self.nan_token = nn.Parameter(torch.randn(2)) ##nan_parameter to be learned, one for each channel
        self.pad_token = nn.Parameter(torch.randn(1, 2))  # shape: (1, C), parameter to be learned for padding


        self.embedding = nn.Linear(self.data_dim, self.d_model)

        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, self.d_model))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, 1, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)

        self.mean_out = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),  # reuse dim feedforward here
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.data_dim),
        )

        self.logvar_out = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),  # reuse dim feedforward here
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.data_dim),
        )

    def num_parameters(self) -> float:
        """Count the number of trainable parameters in the model."""
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = 0
            for p in self.parameters():
                count = 1
                for s in p.size():
                    count *= s
                self._num_parameters += count

        return self._num_parameters
    
    def replace_special_tokens(self, x, pad_val=-9999.0):
        """
        Replace NaNs with self.nan_token and pad_val (e.g. -9999) with self.pad_token.
        x: (B, T, C, H, W)
        """
        leading_dims = x.ndim - 3  # number of leading dims before C

        # Replace NaNs
        if torch.isnan(x).any():
            nan_token = self.nan_token.view(*([1] * leading_dims), -1, 1, 1)
            x = torch.where(torch.isnan(x), nan_token, x)

        # Replace padding 
        if (x == pad_val).any():
            pad_token = self.pad_token.view(*([1] * leading_dims), -1, 1, 1)
            mask = (x == pad_val)
            x = torch.where(mask, pad_token, x)

        return x


    def forward(self, img_baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels, height, width = img_baseline.shape

        #Replace NaNs with learnable parameter
        img_baseline = self.replace_special_tokens(img_baseline)

        assert self.num_patches == (height * width) / (self.patch_size**2)

        img_baseline = einops.rearrange(
            img_baseline, 'b t c (h ph) (w pw) -> b t (h w) (c ph pw)', ph=self.patch_size, pw=self.patch_size
        )  # batch, seq_len, num_patches, data_dim

        img_baseline = (
            self.embedding(img_baseline)
            + self.spatial_pos_embed
            # changed self.temporal_pos_embed[:, :seq_len, :, :]
            # to self.temporal_pos_embed[:, (self.max_seq_len-seq_len):, :, :] to ensure last pre-image always has
            # correct index
            + self.temporal_pos_embed[:, (self.max_seq_len - seq_len) :, :, :]
        )  # batch, seq_len, num_patches, d_model

        img_baseline = img_baseline.view(
            batch_size, seq_len * self.num_patches, self.d_model
        )  # transformer expects (batch_size, sequence, dmodel)

        output = self.transformer_encoder(img_baseline)

        mean = self.mean_out(output)  # batchsize, seq_len*num_patches, data_dim
        logvar = self.logvar_out(output)  # batchsize, seq_len*num_patches, 2*data_dim

        mean = mean.view(batch_size, seq_len, self.num_patches, self.data_dim)  # undo previous operation
        logvar = logvar.view(batch_size, seq_len, self.num_patches, self.data_dim)

        # reshape to be the same shape as input batch_size, seq len, channels, height, width
        mean = einops.rearrange(
            mean,
            'b t (h w) (c ph pw) -> b t c (h ph) (w pw)',
            ph=self.patch_size,
            pw=self.patch_size,
            c=channels,
            h=height // self.patch_size,
            w=width // self.patch_size,
        )

        # reshape so for each pixel we output 4 numbers (ie each entry of cov matrix)
        logvar = einops.rearrange(
            logvar,
            'b t (h w) (c ph pw) -> b t c (h ph) (w pw)',
            ph=self.patch_size,
            pw=self.patch_size,
            c=channels,
            h=height // self.patch_size,
            w=width // self.patch_size,
        )

        return mean[:, -1, ...], logvar[:, -1, ...]
