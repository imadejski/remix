import sys
from collections import OrderedDict
from types import SimpleNamespace

import torch
from gloria.models.text_model import BertEncoder as GLoRIATextEncoder
from gloria.models.vision_model import ImageEncoder as GLoRIAImageEncoder
from PIL import Image
from torch import nn
from torch.nn import functional as F

from remix.utils import GLoRIATransform


class GLoRIAInferenceEngine:
    def __init__(self, model_path: str) -> nn.Module:
        # taken from pretraining config
        # https://github.com/marshuang80/gloria/blob/main/configs/chexpert_pretrain_config.yaml
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                text=SimpleNamespace(
                    bert_type="emilyalsentzer/Bio_ClinicalBERT",
                    last_n_layers=4,
                    aggregate_method="sum",
                    norm=False,
                    embedding_dim=768,
                    freeze_bert=False,
                    agg_tokens=True,
                ),
                vision=SimpleNamespace(
                    model_name="resnet_50",
                    pretrained=True,
                    freeze_cnn=False,
                ),
            ),
            data=SimpleNamespace(
                text=SimpleNamespace(
                    word_num=97,
                ),
            ),
        )
        self.vis = GLoRIAImageEncoder(cfg)
        self.txt = GLoRIATextEncoder(cfg)

        # an interesting note: ImageEncoder upsamples image to 299x299 in the model forward function
        # but the GLoRIATransform implemented in remix does upsample too (since it's weird to put it in a forward function...)
        # - shouldn't issue though, upsample of image at target size shouldn't have a noticeable effect
        self.transform = GLoRIATransform()

        self.tokenizer = self.txt.tokenizer
        self.tokenizer.model_max_length = cfg.data.text.word_num

        # load pretrained model weights
        ckpt = torch.load(model_path, map_location="cpu")
        sd = ckpt["state_dict"]

        vis_sd = OrderedDict()
        txt_sd = OrderedDict()
        for k, v in sd.items():
            if k.startswith("gloria.img_encoder."):
                k = k.replace("gloria.img_encoder.", "")
                vis_sd[k] = v
            elif k.startswith("gloria.text_encoder."):
                k = k.replace("gloria.text_encoder.", "")
                if k == "model.embeddings.position_ids":
                    continue
                txt_sd[k] = v
            else:
                print("Unexpected key:", k, file=sys.stderr)
                continue
        self.vis.load_state_dict(vis_sd, strict=True)
        self.txt.load_state_dict(txt_sd, strict=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vis.eval().to(self.device)
        self.txt.eval().to(self.device)

    def get_image_embed(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        assert image.mode == "L"  # greyscale

        image_tensor = self.transform(image).to(self.device)  #  (C, Wh, Ww)
        assert image_tensor.dim() == 3

        with torch.inference_mode():
            # ImageEncoder forward requires 4D tensor, 0th batch dim
            emb = self.vis(image_tensor.unsqueeze(0), get_local=False)

        emb = emb.squeeze(0)  # (H,)

        return emb

    def get_image_projection(
        self,
        *,  # enforce kwargs
        image_path: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        emb = self.get_image_embed(image_path)

        with torch.inference_mode():
            # ImageEncoder.global_embedder is just a linear layer, don't need batch dim
            proj = self.vis.global_embedder(emb)  # (H',)

        if normalize:
            proj = F.normalize(proj, dim=-1)

        return proj

    def get_text_embed(self, text: str) -> torch.Tensor:
        raise NotImplementedError(
            "Base GLoRIA model does not distinguish between unprojected and "
            "projected embeddings, use get_text_projection instead"
        )

    def get_text_projection(
        self,
        *,  # enforce kwargs
        text: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        ret = self.tokenizer(
            text=text,
            return_tensors="pt",
            padding="longest",  # pad to longest in sequence (no effect with single string...)
            truncation=True,  # but truncate to max model length
        )
        ret = {k: v.to(self.device) for k, v in ret.items()}

        with torch.inference_mode():
            _, proj, _ = self.txt(
                ids=ret["input_ids"],
                attn_mask=ret["attention_mask"],
                token_type=ret["token_type_ids"],
            )
            proj = proj.squeeze()  # (H',)

        if normalize:
            proj = F.normalize(proj, dim=-1)

        return proj
