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
        # Create the conditioning tokens
        input_tokens = self._conditioning_embedding(conditioning_signal)
        assert input_tokens.shape[1] == 1
        assert input_tokens.shape[2] == self.config.model_dim
        batch_size = input_tokens.shape[0]

        num_tokens_to_sample = math.prod(self.config.latent_spatial_dims)

        sampled_latents = []

        for latent_idx in range(num_tokens_to_sample):
            # Add a position embedding to the last input token
            input_tokens[:, -1:] += self._position_embedding[latent_idx].reshape(
                1, 1, -1
            )

            attention_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                input_tokens.shape[1], device=input_tokens.device
            )

            # Compute the next token distribution
            output_tokens = self._transformer(
                input_tokens, mask=attention_mask, is_causal=True
            )

            # Extract the mean and variance
            output_distribution = self._output_projector(output_tokens[:, -1])
            mu = output_distribution[:, : self.config.latent_feature_dim]
            logvar = output_distribution[:, self.config.latent_feature_dim :]
            std = torch.exp(0.5 * logvar)

            # sample the latent
            latent = mu + temperature * std * torch.randn_like(mu)
            assert latent.ndim == 2
            assert latent.shape[0] == batch_size
            assert latent.shape[1] == self.config.latent_feature_dim

            # Add the new latent to the list of input tokens
            sampled_latents.append(latent)
            new_input_tokens = self._input_projector(latent).unsqueeze(1)
            input_tokens = torch.cat([input_tokens, new_input_tokens], dim=1)

        # Collect all of the sampled latents and reshape
        sampled_latents = torch.stack(sampled_latents, dim=-1).reshape(
            batch_size, self.config.latent_feature_dim, *self.config.latent_spatial_dims
        )

        return sampled_latents
