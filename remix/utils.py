import torch
from torch.nn import Upsample
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import BertTokenizer

NORM_T = tuple[float, float, float]


class CXRTokenizer(BertTokenizer):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        temp = pretrained_model_name_or_path
        if "gloria_chexpert_resnet50.ckpt" in pretrained_model_name_or_path:
            temp = "emilyalsentzer/Bio_ClinicalBERT"
        return super().from_pretrained(temp, **kwargs)


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


def make_upsample_transform(upsample: int) -> Upsample:
    # this is buried in the GLoRIA model forward function
    f = Upsample(
        size=(upsample, upsample),
        mode="bilinear",
        align_corners=True,
    )

    def upsample_image(data: torch.Tensor) -> torch.Tensor:
        x = data.unsqueeze(0)
        x = f(x)
        x = x.squeeze()
        return x

    return upsample_image


class ResNet50Transform:
    def __init__(
        self,
        *,  # enforce kwargs
        resize: int,
        center_crop: int,
        normalize: tuple[float, float] | tuple[NORM_T, NORM_T] | None = None,
        upsample: int | None = None,
    ):
        transforms = [
            Resize(resize, antialias=True),
            CenterCrop(center_crop),
            ToTensor(),
            expand_channels,
        ]
        if normalize is not None:
            if isinstance(normalize[0], float):
                normalize = (
                    (normalize[0], normalize[0], normalize[0]),
                    (normalize[1], normalize[1], normalize[1]),
                )
            transforms.append(Normalize(mean=normalize[0], std=normalize[1]))
        if upsample is not None:
            transforms.append(make_upsample_transform(upsample))
        self.transform = Compose(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class BioViLTransform(ResNet50Transform):
    # BioViL and BioViL-T do not normalize
    def __init__(self):
        super().__init__(
            resize=512,
            center_crop=448,
        )


class GLoRIATransform(ResNet50Transform):
    # replicated from GLoRIA source code and config
    def __init__(self):
        super().__init__(
            resize=256,
            center_crop=224,
            normalize=(0.5, 0.5),
            upsample=299,
        )
