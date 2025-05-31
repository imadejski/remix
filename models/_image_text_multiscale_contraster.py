from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from transformers import BertConfig, BertForMaskedLM, BertModel, BertTokenizer

from .modules import LocalGlobalContraster, ResNet50Encoder


class ImageTextMultiScaleContrasterConfig(BertConfig):
    model_type = "image_text_multiscale_contraster"
    attribute_map = {
        "hidden_size": "contraster_t_size",  # assumes text is t and image is i
    }

    def __init__(
        self,
        *,  # enforce kwargs
        contraster_dropout: float = 0.1,
        projection_size: int = 128,
        contraster_temperature: float = 0.07,
        avg_patches_after_proj: bool = True,
        loss_combo: Literal["igl_tgl", "igl_tg", "ig_tgl", "ig_tg"] = "igl_tgl",
        # igl_tgl is image-global/local <--> text-global/local
        # igl_tg is image-global/local <--> text-global
        # ig_tgl is image-global <--> text-global/local
        # ig_tg is image-global <--> text-global
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contraster_dropout = contraster_dropout
        self.contraster_i_size = 2048  # ResNet50
        self.projection_size = projection_size
        self.contraster_temperature = contraster_temperature
        self.avg_patches_after_proj = avg_patches_after_proj
        self.loss_combo = loss_combo


class ImageTextMultiScaleContraster(BertModel):
    config_class = ImageTextMultiScaleContrasterConfig

    def __init__(self, config: ImageTextMultiScaleContrasterConfig):
        super().__init__(config, add_pooling_layer=False)
        self.resnet = ResNet50Encoder(config)
        self.contraster = LocalGlobalContraster(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        *,  # enforce kwargs
        input_ids_global: torch.Tensor,  # (B, T)
        attention_mask_global: torch.Tensor,  # (B, T)
        input_ids_locals: torch.Tensor,  # (B, Lx, T)
        attention_mask_locals: torch.Tensor,  # (B, Lx, T)
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
        text_embed_out = (
            super()
            .forward(input_ids=input_ids, attention_mask=attention_mask)
            .last_hidden_state
        )  # (B + BLx, T, Hx)

        # use 0th token of text sequence (should be [CLS] token embedding)
        text_embed_global = text_embed_out[:B, 0, :]  # (B, Hx)
        text_embed_locals = text_embed_out[B:, 0, :].view(B, Lx, -1)  # (B, Lx, Hx)

        (
            image_embed_locals,  # (B, Hy, Ph, Pw)
            image_embed_global,  # (B, Hy)
        ) = self.resnet(images)
        _, Hy, Ph, Pw = image_embed_locals.shape
        image_embed_locals = image_embed_locals.view(B, Hy, Ph * Pw)  # (B, Hy, Ly=PhPw)
        image_embed_locals = image_embed_locals.mT  # (B, Ly, Hy)

        return self.contraster(
            t_global=text_embed_global,
            t_locals=text_embed_locals,
            i_global=image_embed_global,
            i_locals=image_embed_locals,
        )


class InferenceEngine:
    def __init__(self, model_checkpoint, tokenizer_checkpoint):
        self.config = ImageTextMultiScaleContrasterConfig.from_pretrained(
            model_checkpoint
        )
        self.model = ImageTextMultiScaleContraster.from_pretrained(
            model_checkpoint, config=self.config
        )
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess_image(self, image_path):
        image = read_image(image_path)
        assert image.shape[0] == 1  # greyscale
        image = Image.fromarray(image.numpy()[0]).convert("L")
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]
        )
        image_tensor = transform(image)
        return image_tensor

    def prep_image(self, image_input):
        if isinstance(image_input, str):
            image_tensor = self.preprocess_image(image_input)
        else:
            image_tensor = image_input

        single_image = False
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            single_image = True

        image_tensor = image_tensor.to(self.model.device)

        with torch.no_grad():
            patch_emb, pool_emb = self.model.resnet(image_tensor)

        # Assuming patch_emb has shape (B, C, Wh, Ww)
        batch_size = patch_emb.shape[0]
        num_channels = patch_emb.shape[1]
        height = patch_emb.shape[2]
        width = patch_emb.shape[3]

        # Reshape to (B, C, Wh * Ww)
        patch_emb = patch_emb.view(batch_size, num_channels, -1)

        return patch_emb, height, width, single_image

    def get_projected_patch_embeddings(
        self, image_input, preprocess=True, normalize=True
    ):
        """Returns image patch embeddings, preserved image grid
        input: image tensor (B, C, Wh, Ww), B is batch size, C is number of channels, Wh is height, Ww is width
        """
        patch_emb, height, width, single_image = self.prep_image(image_input)

        # Apply projection to each patch embedding
        projected_patch_embeddings = self.model.contraster.proj_i(
            patch_emb.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # Reshape back to (B, H, Wh, Ww)
        projection_size = self.config.projection_size
        projected_patch_embeddings = projected_patch_embeddings.view(
            patch_emb.shape[0], projection_size, height, width
        )

        if normalize:
            projected_patch_embeddings = F.normalize(projected_patch_embeddings, dim=-1)

        projected_patch_embeddings = projected_patch_embeddings.permute(
            [0, 2, 3, 1]
        )  # (B, H, Wh, Ww) -> (B, Wh, Ww, H)

        if single_image:
            projected_patch_embeddings = projected_patch_embeddings.squeeze(0)

        return projected_patch_embeddings

    def get_projected_global_embeddings(
        self, image_input, preprocess=True, normalize=True, avg_patches_after_proj=True
    ):
        """Returns global image embedding
        input: (B, C, Wh, Ww)"""
        patch_emb, height, width, single_image = self.prep_image(image_input)

        if avg_patches_after_proj:
            # Apply projection to each patch embedding
            projected_patch_embeddings = self.model.contraster.proj_i(
                patch_emb.permute(0, 2, 1)
            ).permute(0, 2, 1)

            # Reshape back to (B, H, Wh, Ww)
            projection_size = self.config.projection_size
            projected_patch_embeddings = projected_patch_embeddings.view(
                patch_emb.shape[0], projection_size, height, width
            )

            # Average the patch embeddings
            projected_global_embeddings = torch.mean(
                projected_patch_embeddings, dim=(2, 3)
            )  # (B, H, Wh, Ww) -> (B, H)
        else:
            # Average the patch embeddings before projection
            global_embedding_pre_proj = torch.mean(
                patch_emb, dim=2
            )  # (B, C, Wh * Ww) -> (B, C)

            # Apply projection to the global embedding
            projected_global_embeddings = self.model.contraster.proj_i(
                global_embedding_pre_proj
            )

        if normalize:
            projected_global_embeddings = F.normalize(
                projected_global_embeddings, dim=-1
            )

        if single_image:
            projected_global_embeddings = projected_global_embeddings.squeeze(0)

        return projected_global_embeddings

    def get_projected_text_embedding(
        self, texts, tokenized=False, normalize=True
    ):  # add a tokenizer bool to determine if input needs to be tokenized
        """Returns text embedding of single text or list of texts"""
        if not tokenized:
            if isinstance(texts, str):
                texts = [texts]

            texts = [text.rstrip("!?.") for text in texts]

            tokenizer_outputs = self.tokenizer(
                text=texts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=True,
            )
            input_ids = tokenizer_outputs.input_ids.to(self.device)
            attention_mask = tokenizer_outputs.attention_mask.to(self.device)

        else:
            input_ids, attention_mask = texts

        with torch.no_grad():
            text_outputs = super(ImageTextMultiScaleContraster, self.model).forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # (B, H)

            if normalize:
                text_embeddings = F.normalize(text_embeddings, dim=-1)

            if len(texts) == 1:
                text_embeddings = text_embeddings.squeeze(0)

            return text_embeddings
