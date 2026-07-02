import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2


def load_dataset(train: bool):
    transforms = v2.Compose(
        [v2.PILToTensor(), v2.Resize((32, 32)), v2.ToDtype(torch.float32, scale=True)]
    )
    return tv.datasets.MNIST(".", download=True, transform=transforms, train=train)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, channel_dims: list[int], latent_dim: int):
        super().__init__()
        self._encoder = GaussianEncoder(
            channel_dims=channel_dims, latent_dim=latent_dim
        )
        self._decoder = GaussianDecoder(
            channel_dims=channel_dims[::-1], latent_dim=latent_dim
        )

    def forward(self, x):
        # Compute the latent mean and variance
        mu, logvar = self._encoder(x)
        std = torch.exp(0.5 * logvar)

        # Sample a latent
        z = std * torch.randn_like(mu) + mu

        # run latents through a decoder
        return mu, logvar, z, self._decoder(z)


def compute_reverse_kl_loss(mu: torch.Tensor, logvar: torch.Tensor, should_reduce=True):
    normalizer_term = logvar
    det_trace_term = torch.exp(logvar)
    mean_term = mu**2

    kl_term = normalizer_term - det_trace_term - mean_term + 1
    if should_reduce:
        return -0.5 * kl_term.sum((1, 2, 3)).mean()
    else:
        return -0.5 * kl_term


def compute_reconstruction_loss(x, x_prime_logit):
    loss = F.binary_cross_entropy_with_logits(x_prime_logit, x, reduction="sum")
    return loss / x.shape[0]


def log_validation_metrics(model, dataloader, writer, batch_size, epoch_idx):
    model.eval()
    num_items = 0
    kl_loss = 0.0
    kl_sum = 0.0
    kl_sum_squares = 0.0
    reconstruction_loss = 0.0
    kl_counts = None
    kl_bins = torch.linspace(0, 5, 101)
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.cuda()
            mu, logvar, _, x_prime_logit = model(imgs)
            raw_kl = (
                compute_reverse_kl_loss(mu, logvar, should_reduce=False)
                .mean(dim=(0))
                .flatten()
            )
            new_counts, _ = torch.histogram(raw_kl.cpu(), kl_bins)
            kl_sum += raw_kl.sum()
            kl_sum_squares = (raw_kl**2).sum()
            kl_counts = new_counts if kl_counts is None else new_counts + kl_counts
            kl_loss += raw_kl.sum()
            reconstruction_loss += compute_reconstruction_loss(imgs, x_prime_logit)
            num_items += imgs.shape[0] / batch_size
    kl_loss = kl_loss / num_items
    reconstruction_loss = reconstruction_loss / num_items

    writer.add_scalar("val/kl_loss", kl_loss.item(), global_step=epoch_idx)
    writer.add_scalar(
        "val/reconstruction_loss", reconstruction_loss.item(), global_step=epoch_idx
    )
    writer.add_images("val/target", imgs[:16], global_step=epoch_idx)
    writer.add_images("val/recon", F.sigmoid(x_prime_logit[:16]), global_step=epoch_idx)
    writer.add_histogram_raw(
        "val/kl_per_dim",
        min=kl_bins[0],
        max=kl_bins[-1],
        num=kl_counts.sum(),
        sum=kl_sum,
        sum_squares=kl_sum_squares,
        bucket_limits=kl_bins[1:],
        bucket_counts=kl_counts,
        global_step=epoch_idx,
    )

    model.train()


def log_train_metrics(kl_loss, reconstruction_loss, writer, total_batch_idx):
    writer.add_scalar("train/kl_loss", kl_loss, global_step=total_batch_idx)
    writer.add_scalar(
        "train/reconstruction_loss",
        reconstruction_loss,
        global_step=total_batch_idx,
    )


def main(
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    kl_factor: float,
    latent_dim: int,
    log_dir: Path,
):
    # Build a dataset
    train_dataset = load_dataset(train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = load_dataset(train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    writer.add_hparams(
        {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "kl_factor": kl_factor,
            "latent_dim": latent_dim,
        },
        {},
    )

    # Build the model
    autoencoder = VariationalAutoEncoder(
        channel_dims=[8, 16, 32], latent_dim=latent_dim
    ).cuda()

    opt = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    total_batch_idx = 0
    for epoch_idx in range(num_epochs):
        for batch_idx, (imgs, _) in enumerate(train_loader):
            total_batch_idx += 1
            opt.zero_grad()
            imgs = imgs.cuda()
            mu, logvar, _, x_prime_logit = autoencoder(imgs)
            kl_loss = compute_reverse_kl_loss(mu, logvar)
            reconstruction_loss = compute_reconstruction_loss(imgs, x_prime_logit)
            loss = kl_factor * kl_loss + reconstruction_loss
            loss.backward()
            opt.step()
            print(
                f"{epoch_idx=} {batch_idx=} loss={loss.detach().item():0.03f}"
                + f" {kl_loss=:0.3f} {reconstruction_loss=:0.3f}",
                end="\r",
            )
            log_train_metrics(
                kl_loss.detach().item(),
                reconstruction_loss.detach().item(),
                writer,
                total_batch_idx,
            )
        print()
        log_validation_metrics(
            model=autoencoder,
            dataloader=test_loader,
            writer=writer,
            batch_size=batch_size,
            epoch_idx=epoch_idx,
        )

        if epoch_idx % 5 == 0:
            torch.save(
                autoencoder.state_dict(), log_dir / f"autoencoder_{epoch_idx:03d}.pt"
            )

    torch.save(autoencoder.state_dict(), log_dir / f"autoencoder_{epoch_idx:03d}.pt")
    # Train the model
    print("Hello from vae!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--kl_factor", type=float, default=1.0)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--log_dir", required=True)
    args = parser.parse_args()
    main(
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        args.kl_factor,
        args.latent_dim,
        Path(args.log_dir),
    )
