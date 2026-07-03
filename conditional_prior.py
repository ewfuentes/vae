import math
from dataclasses import dataclass

import msgspec
import torch


class ConditionalPriorConfig(msgspec.Struct):
    latent_spatial_dims: list[int]
    latent_feature_dim: int
    model_dim: int
    num_attention_heads: int
    num_layers: int


@dataclass
class ConditioningSignal:
    digit_class: torch.Tensor


@dataclass
class GaussianLatentPrediction:
    mu: torch.Tensor
    logvar: torch.Tensor


class ConditioningEmbedding(torch.nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        NUM_DIGITS = 10
        self.digit_embedding = torch.nn.Embedding(NUM_DIGITS, model_dim)

    def forward(self, conditioning: ConditioningSignal) -> torch.Tensor:
        return self.digit_embedding(conditioning.digit_class).unsqueeze(1)


class ConditionalPrior(torch.nn.Module):
    def __init__(self, config: ConditionalPriorConfig):
        super().__init__()
        self.config = config

        self._conditioning_embedding = ConditioningEmbedding(self.config.model_dim)

        num_position_embeddings = math.prod(self.config.latent_spatial_dims)
        self._position_embedding = torch.nn.Parameter(
            torch.randn((num_position_embeddings, self.config.model_dim))
        )

        self._input_projector = torch.nn.Sequential(
            torch.nn.Linear(self.config.latent_feature_dim, self.config.model_dim),
            torch.nn.LayerNorm(self.config.model_dim),
        )

        self._output_projector = torch.nn.Linear(
            self.config.model_dim, 2 * self.config.latent_feature_dim
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.config.model_dim,
            nhead=self.config.num_attention_heads,
            dim_feedforward=4 * self.config.model_dim,
            batch_first=True,
            norm_first=True,
        )
        self._transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_layers, enable_nested_tensor=False
        )

    def forward(
        self, conditioning_signal: torch.Tensor, sampled_latents: torch.Tensor
    ) -> GaussianLatentPrediction:
        # Convert the conditioning signal into an embedded token (batch x 1 x model dim)
        conditioning_tokens = self._conditioning_embedding(conditioning_signal)
        assert conditioning_tokens.shape[1] == 1
        assert conditioning_tokens.shape[2] == self.config.model_dim

        # Flatten the latent grid [B, C, H, W] -> [B, H*W, C] (fixes a raster order)
        n_batch, n_channel, n_rows, n_cols = sampled_latents.shape
        assert n_channel == self.config.latent_feature_dim
        assert n_rows == self.config.latent_spatial_dims[0]
        assert n_cols == self.config.latent_spatial_dims[1]
        sampled_latents = sampled_latents.permute(0, 2, 3, 1).reshape(
            (n_batch, n_rows * n_cols, n_channel)
        )
        # Drop the last latent
        sampled_latents = sampled_latents[:, :-1, :]

        # Project the latents into tokens
        latent_tokens = self._input_projector(sampled_latents)

        # Assemble the sequence: [conditioning token] ++ latent tokens
        input_tokens = torch.cat([conditioning_tokens, latent_tokens], dim=1)

        # Add position embeddings
        input_tokens += self._position_embedding.unsqueeze(0)

        # Create a causal attention mask
        attention_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            input_tokens.shape[1], device=input_tokens.device
        )

        # Run it through the transformer decoder
        output_tokens = self._transformer(
            input_tokens, mask=attention_mask, is_causal=True
        )

        # Compute the mean/variance predictions from the per-position outputs.
        output_distribution = self._output_projector(output_tokens)
        output_distribution = output_distribution.permute(0, 2, 1).reshape(
            n_batch, 2 * n_channel, n_rows, n_cols
        )
        mu = output_distribution[:, : self.config.latent_feature_dim]
        logvar = output_distribution[:, self.config.latent_feature_dim :]
        return GaussianLatentPrediction(mu=mu, logvar=logvar)

    @torch.no_grad()
    def generate(
        self,
        conditioning_signal: ConditioningSignal,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # Inference-time ancestral sampling (NOT teacher forced): the sequence
        # starts as just the conditioning token and grows one latent at a time.
        #
        # Embed the conditioning signal -> the first (position 0) token.
        # Let N = prod(latent_spatial_dims) be the number of latents to sample.
        # For step i in range(N):
        #   - Add position embeddings to the current (i+1)-length sequence
        #   - Run it through the transformer decoder with a causal mask
        #     (a KV cache is optional; negligible for N this small)
        #   - Take the LAST position's output -> output projector -> (mu, logvar)
        #   - Sample z_i = mu + temperature * exp(0.5 * logvar) * randn_like(mu)
        #     (temperature < 1 = less diverse / sharper; > 1 = more diverse)
        #   - Project z_i back to model_dim and APPEND it as the next input token
        # Stack the sampled z_i and reshape [B, N, C] -> the latent grid
        # [B, C, H, W] so it can be handed straight to the VAE decoder.
        ...
