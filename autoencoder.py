import msgspec
import torch


class VariationalAutoEncoderConfig(msgspec.Struct):
    channel_dims: list[int] = msgspec.field(default_factory=lambda: [8, 16, 32])
    latent_dim: int = 16


class GaussianEncoder(torch.nn.Module):
    def __init__(self, channel_dims=None, latent_dim=16):
        super().__init__()
        self._latent_dim = latent_dim
        if channel_dims is None:
            channel_dims = [8, 16, 32]
        channels = [1] + channel_dims
        layers = []
        for idx in range(len(channels) - 1):
            in_channels = channels[idx]
            out_channels = channels[idx + 1]
            layers += [
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                ),
                torch.nn.GroupNorm(4, out_channels),
                torch.nn.ReLU(),
            ]
        layers += [
            torch.nn.Conv2d(channels[-1], 2 * latent_dim, kernel_size=1, stride=1),
            torch.nn.GroupNorm(4, 2 * latent_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2 * latent_dim, 2 * latent_dim, kernel_size=1, stride=1),
        ]

        self._layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._layers(x)
        mu = out[:, : self._latent_dim]
        logvar = out[:, self._latent_dim :]
        return mu, logvar


class GaussianDecoder(torch.nn.Module):
    def __init__(self, channel_dims=None, latent_dim=16):
        super().__init__()
        if channel_dims is None:
            channel_dims = [32, 16, 8]
        channels = [latent_dim] + channel_dims

        layers = []
        for idx in range(len(channels) - 1):
            in_channels = channels[idx]
            out_channels = channels[idx + 1]
            layers += [
                torch.nn.Upsample(scale_factor=2),
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.GroupNorm(4, out_channels),
                torch.nn.ReLU(),
            ]
        layers += [
            torch.nn.Conv2d(channels[-1], 1, kernel_size=1, stride=1),
        ]

        self._layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self._layers(x)


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, config: VariationalAutoEncoderConfig):
        super().__init__()
        self.config = config

        self._encoder = GaussianEncoder(
            channel_dims=self.config.channel_dims, latent_dim=self.config.latent_dim
        )
        self._decoder = GaussianDecoder(
            channel_dims=self.config.channel_dims[::-1],
            latent_dim=self.config.latent_dim,
        )

    def sample_latents(self, x):
        # Compute the latent mean and variance
        mu, logvar = self._encoder(x)
        std = torch.exp(0.5 * logvar)

        # Sample a latent
        z = std * torch.randn_like(mu) + mu
        return mu, logvar, z

    def forward(self, x):
        # sample latents
        mu, logvar, z = self.sample_latents(x)

        # run latents through a decoder
        return mu, logvar, z, self._decoder(z)

    def save(self, path):
        # Store the config as plain builtins so the checkpoint stays loadable
        # under torch.load(weights_only=True).
        torch.save(
            {
                "config": msgspec.to_builtins(self.config),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        config = msgspec.convert(ckpt["config"], VariationalAutoEncoderConfig)
        model = cls(config)
        model.load_state_dict(ckpt["state_dict"])
        return model
