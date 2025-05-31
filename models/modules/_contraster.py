from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig


class Contraster(nn.Module):
    """
    Models which use this module should include the following in their config:
    * contraster_dropout: float = dropout rate
    * contraster_x_size: int = input dimension for x
    * contraster_y_size: int = input dimension for y
    * projection_size: int = contrastive dimension
    * contraster_temperature: float = temperature
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.contraster_dropout)
        self.proj_x = nn.Linear(config.contraster_x_size, config.projection_size)
        self.proj_y = nn.Linear(config.contraster_y_size, config.projection_size)
        self.temperature = config.contraster_temperature

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        y = self.dropout(y)

        x = self.proj_x(x)
        y = self.proj_y(y)

        x = F.normalize(x)
        y = F.normalize(y)

        logits_x = (x @ y.T) * self.temperature
        logits_y = (y @ x.T) * self.temperature

        targets = torch.arange(len(x), device=x.device)
        loss_x = F.cross_entropy(logits_x, targets)
        loss_y = F.cross_entropy(logits_y, targets)

        loss = (loss_x + loss_y) / 2
        return loss


class LocalGlobalContraster(nn.Module):
    """
    Models which use this module should include the following in their config:
    * contraster_dropout: float = dropout rate
    * contraster_t_size: int = input dimension for t (text)
    * contraster_i_size: int = input dimension for i (image)
    * projection_size: int = contrastive dimension
    * contraster_temperature: float = temperature parameter (scalar)
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.contraster_dropout)
        self.proj_t = nn.Linear(config.contraster_t_size, config.projection_size)
        self.proj_i = nn.Linear(config.contraster_i_size, config.projection_size)
        self.temperature = config.contraster_temperature
        self.avg_patches_after_proj = config.avg_patches_after_proj
        self.loss_combo = config.loss_combo

    def info_nce(self, r, s):
        # Determine the dimensions of r and s
        if r.dim() == 3:
            B, L_r, H_r = r.shape
            r = r.view(-1, H_r)
        elif r.dim() == 2:
            B, H_r = r.shape
            L_r = 1
            r = r.view(-1, H_r)

        if s.dim() == 3:
            B, L_s, H_s = s.shape
            s = s.view(-1, H_s)
        elif s.dim() == 2:
            B, H_s = s.shape
            L_s = 1
            s = s.view(-1, H_s)

        # calculate cosine similarity between all image-text pairs
        similarities = torch.einsum("rh,sh->rs", r, s)
        similarities = similarities / self.temperature

        # Average for every group of L_s so that the shape is (B * L_r, B)
        new_shape = (B * L_r, B, L_s)
        reshaped_similarities = similarities.view(new_shape)
        avg_similarities = reshaped_similarities.mean(dim=-1)

        targets = (
            torch.arange(B).repeat_interleave(L_r).to(r.device)
        )  # Shape: (B * L_r)

        logits = avg_similarities

        loss = F.cross_entropy(logits, targets)

        return loss

    def forward(
        self,
        *,  # enforce kwargs
        t_global: torch.Tensor,  # (B, H_t), global text input
        i_global: torch.Tensor,  # (B, H_i), global image input
        t_locals: torch.Tensor,  # (B, L_t, H_t) local text input
        i_locals: torch.Tensor,  # (B, L_i, H_i) or (B, L_i, W_i, H_i), local image input
    ) -> torch.Tensor:
        assert t_global.dim() == 2, "Global text embedding is not 2 dimensional"
        assert i_global.dim() == 2, "Global image embedding is not 2 dimensional"
        assert t_locals.dim() == 3, "Local text embedding is not 3 dimensional"
        assert (
            i_locals.dim() == 3
        ), "Local image embedding is not 3 dimensional"  # image locals flattened before contraster

        t_global = self.dropout(t_global)
        t_locals = self.dropout(t_locals)
        i_locals = self.dropout(i_locals)

        t_global = self.proj_t(t_global)  # note proj_t is for text projection
        t_locals = self.proj_t(t_locals)
        i_locals = self.proj_i(i_locals)  # note proj_i is for image projection

        # take average of patch embeddings after projection layer, if necessary
        if self.avg_patches_after_proj == False:
            i_global = self.dropout(i_global)
            i_global = self.proj_i(i_global)
        else:
            i_global = i_locals.mean(
                dim=(1)
            )  # take average of middle dimension of 3d patch embedding

        # Normalize
        t_global = F.normalize(t_global, dim=-1)
        i_global = F.normalize(i_global, dim=-1)
        t_locals = F.normalize(t_locals, dim=-1)
        i_locals = F.normalize(i_locals, dim=-1)

        # Return combined loss of correct loss calculation combination
        if self.loss_combo == "igl_tgl":  # image-global/local <--> text-global/local
            loss = (
                self.info_nce(t_locals, i_locals)
                + self.info_nce(i_locals, t_locals)
                + self.info_nce(t_locals, i_global)
                + self.info_nce(i_locals, t_global)
                + self.info_nce(t_global, i_locals)
                + self.info_nce(i_global, t_locals)
                + self.info_nce(t_global, i_global)
                + self.info_nce(i_global, t_global)
            ) / 8
        elif self.loss_combo == "igl_tg":  # image-global/local <--> text-global
            loss = (
                self.info_nce(i_locals, t_global)
                + self.info_nce(t_global, i_locals)
                + self.info_nce(t_global, i_global)
                + self.info_nce(i_global, t_global)
            ) / 4
        elif self.loss_combo == "ig_tgl":  # image-global <--> text-global/local
            loss = (
                self.info_nce(t_locals, i_global)
                + self.info_nce(i_global, t_locals)
                + self.info_nce(t_global, i_global)
                + self.info_nce(i_global, t_global)
            ) / 4
        elif self.loss_combo == "ig_tg":  # image-global <--> text-global
            loss = (
                self.info_nce(t_global, i_global) + self.info_nce(i_global, t_global)
            ) / 2
        else:
            raise ValueError("loss_combo is not defined")

        return loss
