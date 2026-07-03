import torch
import torchvision as tv
from torchvision.transforms import v2


def load_dataset(train: bool):
    transforms = v2.Compose(
        [v2.PILToTensor(), v2.Resize((32, 32)), v2.ToDtype(torch.float32, scale=True)]
    )
    return tv.datasets.MNIST(".", download=True, transform=transforms, train=train)
