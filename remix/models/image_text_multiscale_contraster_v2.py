from collections import OrderedDict
from typing import Literal

import torch
from transformers import BertConfig, BertForMaskedLM

from .modules import LocalGlobalContrasterV2, ResNet50Encoder

ALIGNMENT_T = Literal[
    "igl_tgl",  # igl_tgl is image-global/local <--> text-global/local
    "igl_tg",  # igl_tg is image-global/local <--> text-global
    "ig_tgl",  # ig_tgl is image-global <--> text-global/local
    "ig_tg",  # ig_tg is image-global <--> text-global
]


class ImageTextMultiScaleContrasterV2Config(BertConfig):
    model_type = "image_text_multiscale_contraster"
    attribute_map = {
        "hidden_size": "contraster_t_size",  # assumes text is t and image is i
    }

    def __init__(
        self,
        *,  # enforce kwargs
        contraster_dropout: float = 0.1,
        projection_size: int = 128,
        projection_depth: int = 3,
        loss_combo: ALIGNMENT_T = "ig_tgl",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contraster_dropout = contraster_dropout
        self.contraster_i_size = 2048  # ResNet50
        self.projection_size = projection_size
        self.projection_depth = projection_depth
        self.loss_combo = loss_combo

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        temp = pretrained_model_name_or_path
        if "gloria_chexpert_resnet50.ckpt" in pretrained_model_name_or_path:
            temp = "emilyalsentzer/Bio_ClinicalBERT"
        return super().from_pretrained(temp, **kwargs)


class ImageTextMultiScaleContrasterV2(BertForMaskedLM):
    config_class = ImageTextMultiScaleContrasterV2Config

    def __init__(self, config: ImageTextMultiScaleContrasterV2Config):
        config.output_hidden_states = True
        # No pooling layer on base BertModel
        super().__init__(config)
        self.resnet = ResNet50Encoder(config)
        self.contraster = LocalGlobalContrasterV2(config)

        # Initialize weights and apply final processing
        self.post_init()
        # should not modify initial temperature
        assert self.contraster.temperature == 0.5

    def forward(
        self,
        *,  # enforce kwargs
        input_ids_global: torch.Tensor,  # (B, T)
        attention_mask_global: torch.Tensor,  # (B, T)
        labels_global: torch.Tensor | None = None,  # (B, T)
        input_ids_locals: torch.Tensor,  # (B, Lx, T)
        attention_mask_locals: torch.Tensor,  # (B, Lx, T)
        local_mask: torch.Tensor,  # (B, Lx)
        images: torch.Tensor,  # (B, C, Wh, Ww),
    ) -> torch.Tensor:
        B, Lx, T = input_ids_locals.shape
        input_ids = torch.concat(
            [
                input_ids_global,
                input_ids_locals.view(-1, T),
            ]
        )  # (B + BLx, T)
        attention_mask = torch.concat(
            [
                attention_mask_global,
                attention_mask_locals.view(-1, T),
            ]
        )
        labels = None
        if labels_global is not None:
            dummies = torch.ones(
                B * Lx,
                T,
                dtype=labels_global.dtype,
                device=labels_global.device,
            )
            labels = torch.concat(
                [
                    labels_global,
                    dummies * -100,
                    # ignore local MLM ^
                ],
            )
        bert_mlm_out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # get last hidden state of underlying bert model
        text_embed_out = bert_mlm_out.hidden_states[-1]  # (B + BLx, T, Hx)

        # use 0th token of text sequence ([CLS] token embedding)
        text_embed_global = text_embed_out[:B, 0, :]  # (B, Hx)
        text_embed_locals = text_embed_out[B:, 0, :].view(B, Lx, -1)  # (B, Lx, Hx)

        (
            image_embed_locals,  # (B, Hy, Ph, Pw)
            image_embed_global,  # (B, Hy)
        ) = self.resnet(images)
        _, Hy, Ph, Pw = image_embed_locals.shape
        image_embed_locals = image_embed_locals.view(B, Hy, Ph * Pw)  # (B, Hy, Ly=PhPw)
        image_embed_locals = image_embed_locals.mT  # (B, Ly, Hy)

        contrastive_loss = self.contraster(
            t_global=text_embed_global,
            t_locals=text_embed_locals,
            t_local_mask=local_mask,
            i_global=image_embed_global,
            i_locals=image_embed_locals,
        )
        if bert_mlm_out.loss is not None:
            return contrastive_loss + bert_mlm_out.loss
        return contrastive_loss

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        temp = pretrained_model_name_or_path
        # if loading base gloria checkpoint, use base bert model used by
        # gloria authors for configuration and load bert state dict after
        if "gloria_chexpert_resnet50.ckpt" in pretrained_model_name_or_path:
            temp = "emilyalsentzer/Bio_ClinicalBERT"

        model = super().from_pretrained(temp, **kwargs)

        # initialize image model from pretrained weights
        if pretrained_model_name_or_path == "microsoft/BiomedVLP-BioViL-T":
            model.resnet.load_biovil_t_weights()
        elif (
            pretrained_model_name_or_path == "microsoft/BiomedVLP-CXR-BERT-specialized"
        ):
            model.resnet.load_biovil_weights()
        elif "gloria_chexpert_resnet50.ckpt" in pretrained_model_name_or_path:
            model.load_gloria_weights(pretrained_model_name_or_path)
            model.resnet.load_gloria_weights(pretrained_model_name_or_path)
        else:
            print(
                "Loading pretrained model from non-base model, "
                "check that both image and text encoders are loaded correctly"
            )
        return model

    def load_gloria_weights(self, gloria_checkpoint_path: str) -> None:
        ckpt = torch.load(gloria_checkpoint_path, map_location="cpu")
        sd = ckpt["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            k = k.replace("gloria.text_encoder.model.", "bert.")
            if k.startswith("gloria"):
                continue
            new_state_dict[k] = v
        incompatible_keys = self.load_state_dict(
            new_state_dict,
            strict=False,
        )

        print("Loaded GLoRIA BERT weights, missing the following keys:")
        for k in incompatible_keys.missing_keys:
            if k.startswith("resnet.") or k.startswith("contraster."):
                # expected that these keys won't be in sd
                continue
            print(k)
        print()
        print("Loaded GLoRIA BERT weights, did not expect the following keys:")
        for k in incompatible_keys.unexpected_keys:
            print(k)
        print()
