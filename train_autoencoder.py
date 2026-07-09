import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: build figures without a display

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import ConfusionMatrix
from torchmetrics.image.kid import KernelInceptionDistance

from autoencoder import VariationalAutoEncoder, VariationalAutoEncoderConfig
from classifier import Classifier
from dataset import load_dataset


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


def log_validation_metrics(
    model, classifier, dataloader, kid, writer, batch_size, epoch_idx
):
    model.eval()
    num_batches = 0
    num_items = 0
    match_count = 0
    kl_loss = 0.0
    kl_sum = 0.0
    kl_sum_squares = 0.0
    reconstruction_loss = 0.0
    kl_counts = None
    kl_bins = torch.linspace(0, 5, 101)
    # rows = class of the real image, cols = class of its reconstruction
    confusion_matrix = ConfusionMatrix(
        task="multiclass", num_classes=10, normalize="true"
    )
    with torch.no_grad():
        kid.reset()
        for imgs, _ in dataloader:
            imgs = imgs.cuda()
            mu, logvar, _, x_prime_logit = model(imgs)

            kid.update(F.sigmoid(x_prime_logit), real=False)
            image_preds = classifier(imgs)
            xprime_preds = classifier(F.sigmoid(x_prime_logit))
            image_preds = torch.argmax(image_preds, dim=-1)
            xprime_preds = torch.argmax(xprime_preds, dim=-1)
            match_count += (image_preds == xprime_preds).sum()
            num_items += imgs.shape[0]
            confusion_matrix.update(xprime_preds.cpu(), image_preds.cpu())

            raw_kl = (
                compute_reverse_kl_loss(mu, logvar, should_reduce=False)
                .mean(dim=(0))
                .flatten()
            )
            new_counts, _ = torch.histogram(raw_kl.cpu(), kl_bins)
            kl_sum += raw_kl.sum()
            kl_sum_squares += (raw_kl**2).sum()
            kl_counts = new_counts if kl_counts is None else new_counts + kl_counts
            kl_loss += raw_kl.sum()
            reconstruction_loss += compute_reconstruction_loss(imgs, x_prime_logit)
            num_batches += imgs.shape[0] / batch_size
    kl_loss = kl_loss / num_batches
    reconstruction_loss = reconstruction_loss / num_batches

    writer.add_scalar(
        "val/classification_accuracy", match_count / num_items, global_step=epoch_idx
    )
    writer.add_scalar("val/kl_loss", kl_loss, global_step=epoch_idx)
    writer.add_scalar(
        "val/reconstruction_loss", reconstruction_loss, global_step=epoch_idx
    )
    kid_mean, kid_std = kid.compute()
    writer.add_scalar("val/kid_mean", kid_mean, global_step=epoch_idx)
    writer.add_scalar("val/kid_std", kid_std, global_step=epoch_idx)
    writer.add_images("val/target", imgs[:16], global_step=epoch_idx)
    writer.add_images("val/recon", F.sigmoid(x_prime_logit[:16]), global_step=epoch_idx)
    cm_fig = plt.figure(figsize=(8, 8))
    cm_fig, _ = confusion_matrix.plot(ax=plt.gca())
    writer.add_figure("val/confusion_matrix", cm_fig, global_step=epoch_idx)
    plt.close(cm_fig)
    writer.add_histogram_raw(
        "val/kl_per_dim",
        min=kl_bins[0],
        max=kl_bins[-1],
        num=0 if kl_counts is None else kl_counts.sum(),
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


def create_kid_object(dataloader, classifier):
    class _Features(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self._model = model

        def forward(self, x):
            return self._model.features(x)

    kid = KernelInceptionDistance(
        feature=_Features(classifier), reset_real_features=False, normalize=True
    )
    with torch.no_grad():
        for imgs, _ in dataloader:
            kid.update(imgs.cuda(), real=True)
    return kid


def main(
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    kl_factor: float,
    latent_dim: int,
    log_dir: Path,
    classifier_path: Path,
):
    # Load the classifier for validation metrics
    mnist_classifier = Classifier.load(classifier_path).cuda()

    # Build a dataset
    train_dataset = load_dataset(train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = load_dataset(train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    kid = create_kid_object(test_loader, mnist_classifier)

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
    config = VariationalAutoEncoderConfig(
        channel_dims=[8, 16, 32], latent_dim=latent_dim
    )
    autoencoder = VariationalAutoEncoder(config).cuda()

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
            classifier=mnist_classifier,
            dataloader=test_loader,
            kid=kid,
            writer=writer,
            batch_size=batch_size,
            epoch_idx=epoch_idx,
        )

        if epoch_idx % 5 == 0:
            autoencoder.save(log_dir / f"autoencoder_{epoch_idx:03d}.pt")

    autoencoder.save(log_dir / f"autoencoder_{epoch_idx:03d}.pt")
    # Train the model
    print("Hello from vae!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--kl_factor", type=float, default=1.0)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--classifier_path", required=True)
    args = parser.parse_args()
    main(
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        args.kl_factor,
        args.latent_dim,
        Path(args.output_dir),
        Path(args.classifier_path),
    )
