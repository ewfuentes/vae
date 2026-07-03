import argparse
from pathlib import Path

import torch

from autoencoder import VariationalAutoEncoder
from conditional_prior import (
    ConditionalPrior,
    ConditionalPriorConfig,
    ConditioningSignal,
)
from dataset import load_dataset


def load_autoencoder(autoencoder_path: Path) -> VariationalAutoEncoder:
    return VariationalAutoEncoder.load(autoencoder_path)


def compute_kl_loss(
    pred_mu: torch.Tensor,
    pred_logvar: torch.Tensor,
    target_mu: torch.Tensor,
    target_logvar: torch.Tensor,
):
    normalizer_term = pred_logvar - target_logvar
    det_trace_term = torch.exp(target_logvar - pred_logvar)
    dimension_term = 1
    mean_term = (target_mu - pred_mu) ** 2 * torch.exp(-pred_logvar)
    kl_divergence = 0.5 * (
        normalizer_term - dimension_term + det_trace_term + mean_term
    )
    kl_loss = kl_divergence.sum((1, 2, 3)).mean()
    return kl_loss


def main(
    autoencoder_path: Path,
    output_dir: Path,
    model_dim: int,
    num_attention_heads: int,
    num_layers: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
):
    # Load the dataset
    train_dataset = load_dataset(train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Load the autoencoder and pass an image through to learn
    # latent dimension
    autoencoder = load_autoencoder(autoencoder_path).cuda()
    img = next(iter(train_dataloader))[0].cuda()
    _, _, z, _ = autoencoder(img)

    # Create the conditional learned prior
    prior = ConditionalPrior(
        ConditionalPriorConfig(
            latent_spatial_dims=z.shape[-2:],
            latent_feature_dim=z.shape[-3],
            model_dim=model_dim,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
        )
    ).cuda()

    opt = torch.optim.Adam(prior.parameters(), lr=learning_rate)

    for epoch_idx in range(num_epochs):
        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            with torch.no_grad():
                target_mu, target_logvar, z = autoencoder.sample_latents(imgs.cuda())

            opt.zero_grad()

            # Give the labels and the sampled latents to the conditional prior
            pred = prior(ConditioningSignal(digit_class=labels.cuda()), z)

            # compute kl divergence between encoder and conditional prior
            loss = compute_kl_loss(
                pred_mu=pred.mu,
                pred_logvar=pred.logvar,
                target_mu=target_mu,
                target_logvar=target_logvar,
            )
            loss.backward()
            opt.step()

            print(
                f"{epoch_idx=} {batch_idx=} loss={loss.detach().item():0.3f}", end="\r"
            )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    args = parser.parse_args()

    main(
        autoencoder_path=Path(args.autoencoder_path),
        output_dir=Path(args.output_dir),
        model_dim=args.model_dim,
        num_attention_heads=args.num_attention_heads,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
