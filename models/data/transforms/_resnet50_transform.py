import torch
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor


def expand_channels(data: torch.Tensor) -> torch.Tensor:
    """
    :param data: Tensor of shape [1, H, W].
    :return: Tensor with channel copied three times, shape [3, H, W].
    """
    if data.shape[0] == 1:
        return torch.repeat_interleave(data, 3, dim=0)
    elif data.shape[0] == 3:
        return data
    raise ValueError(f"Expected input of shape [1 or 3, H, W], found {data.shape}")


class ResNet50Transform:
    # TODO normalize?
    def __init__(self, resize: int = 512, center_crop: int = 448):
        self.transform = Compose(
            [
                Resize(resize, antialias=True),
                CenterCrop(center_crop),
                ToTensor(),
                expand_channels,
            ]
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
