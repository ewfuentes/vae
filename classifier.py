import msgspec
import torch
import torch.nn.functional as F


class ClassifierConfig(msgspec.Struct):
    channel_dims: list[int] | None
    out_dim: int


class Classifier(torch.nn.Module):
    def __init__(self, config: ClassifierConfig):
        super().__init__()

        self.config = config

        if self.config.channel_dims is None:
            self.config.channel_dims = [8, 16, 32, 64]

        channel_dims = [1] + self.config.channel_dims
        layers = []
        for idx in range(len(channel_dims) - 1):
            in_channels = channel_dims[idx]
            out_channels = channel_dims[idx + 1]
            layers += [
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, stride=2
                ),
                torch.nn.GroupNorm(4, out_channels),
                torch.nn.ReLU(),
            ]

        layers += [
            torch.nn.Conv2d(
                channel_dims[-1], channel_dims[-1], kernel_size=1, stride=1
            ),
            torch.nn.GroupNorm(4, channel_dims[-1]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                channel_dims[-1], channel_dims[-1], kernel_size=1, stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(self.config.out_dim),
        ]

        self._model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

    def save(self, path):
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
        config = msgspec.convert(ckpt["config"], ClassifierConfig)
        model = cls(config)
        model.load_state_dict(ckpt["state_dict"])
        return model
