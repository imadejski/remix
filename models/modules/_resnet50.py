from collections import OrderedDict

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models.resnet import Bottleneck, ResNet
from transformers import PretrainedConfig

BIOVIL_T_URL = "https://huggingface.co/microsoft/BiomedVLP-BioViL-T/resolve/v1.0/biovil_t_image_model_proj_size_128.pt"
BIOVIL_URL = "https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized/resolve/main/biovil_image_resnet50_proj_size_128.pt"


class ResNet50Encoder(ResNet):
    def __init__(self, config: PretrainedConfig):
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3])
        del self.fc
        del self.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        patch_emb = self.layer4(x3)

        pooled = F.adaptive_avg_pool2d(patch_emb, (1, 1)).flatten(start_dim=1)

        return patch_emb, pooled

    def _load_biovil_weights(self, url):
        state_dict = load_state_dict_from_url(
            url,
            map_location="cpu",
        )
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("encoder.encoder."):
                k = k.replace("encoder.encoder.", "")
                if k.startswith("fc."):
                    continue
            else:
                continue
            new_state_dict[k] = v
        self.load_state_dict(
            new_state_dict,
            strict=True,
        )

    def load_biovil_weights(self):
        self._load_biovil_weights(BIOVIL_URL)

    def load_biovil_t_weights(self):
        self._load_biovil_weights(BIOVIL_T_URL)
