from pathlib import Path

import torch
import torch.nn.functional as F
import torchmetrics.functional.classification

from classifier import Classifier, ClassifierConfig
from dataset import load_dataset


def main(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_dataset(train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=1024
    )

    validation_dataset = load_dataset(train=False)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1024
    )

    num_epochs = 25

    model = Classifier(
        ClassifierConfig(channel_dims=[8, 16, 32, 64], out_dim=10)
    ).cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch_idx in range(num_epochs):
        for batch_idx, (img, label) in enumerate(train_dataloader):
            opt.zero_grad()
            logits = model(img.cuda())
            loss = F.cross_entropy(logits, label.cuda())
            loss.backward()
            opt.step()

            print(
                f"{epoch_idx=} {batch_idx=} loss={loss.detach().item():0.3f}", end="\r"
            )
        print()

        # compute validation loss
        output_preds = []
        output_labels = []
        model.eval()
        with torch.no_grad():
            for img, label in validation_dataloader:
                output_preds.append(F.softmax(model(img.cuda()), dim=-1).cpu())
                output_labels.append(label)

        output_preds = torch.cat(output_preds, dim=0)
        output_labels = torch.cat(output_labels, dim=0)
        auroc = torchmetrics.functional.classification.multiclass_auroc(
            output_preds, output_labels, num_classes=10
        )
        accuracy = torchmetrics.functional.classification.multiclass_accuracy(
            output_preds, output_labels, num_classes=10
        )
        print(f"{auroc=:0.3f} {accuracy=:0.3f}")

        model.train()

    # Save the final model.
    model.save(output_dir / "classifier.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    main(Path(args.output_dir))
