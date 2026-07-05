from pathlib import Path

import torch
import torch.nn.functional as F

from classifier import Classifier, ClassifierConfig
from dataset import load_dataset


def main(output_dir: Path):
    train_dataset = load_dataset(train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=1024
    )

    test_dataset = load_dataset(train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

    num_epochs = 100

    model = Classifier(
        ClassifierConfig(channel_dims=[8, 16, 32, 64], out_dim=10)
    ).cuda()

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    main(args.output_dir)
