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
    * avg_patches_after_proj: bool = if True, projected global image embedding is the average of the projected local patch embeddings
    * loss_combo: Literal["ig_tg", "ig_tgl", "igl_tg", "igl_tgl"] = contrastive pattern
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
        i_locals: torch.Tensor,  # (B, L_i, H_i) local image input
    ) -> torch.Tensor:
        assert t_global.dim() == 2, "Global text embedding is not 2 dimensional"
        assert i_global.dim() == 2, "Global image embedding is not 2 dimensional"
        assert t_locals.dim() == 3, "Local text embedding is not 3 dimensional"
        assert i_locals.dim() == 3, "Local image embedding is not 3 dimensional"  # image locals not flattened before contraster? # fmt: skip

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


def make_mlp(
    *,  # enforce kwargs
    input_size: int,
    output_size: int,
    depth: int,
) -> nn.Sequential:
    assert depth > 0
    layers = [nn.Linear(input_size, output_size)]
    layers += [nn.Linear(output_size, output_size) for _ in range(depth - 1)]
    return nn.Sequential(*layers)


class LocalGlobalContrasterV2(nn.Module):
    """
    Models which use this module should include the following in their config:
    * contraster_dropout: float = dropout rate
    * contraster_t_size: int = input dimension for t (text)
    * contraster_i_size: int = input dimension for i (image)
    * projection_size: int = contrastive dimension
    * projection_depth: int = depth of MLP projectors
    * loss_combo: Literal["ig_tg", "ig_tgl", "igl_tg", "igl_tgl"] = contrastive pattern
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.contraster_dropout)
        self.proj_t = make_mlp(
            input_size=config.contraster_t_size,
            output_size=config.projection_size,
            depth=config.projection_depth,
        )
        self.proj_i = make_mlp(
            input_size=config.contraster_i_size,
            output_size=config.projection_size,
            depth=config.projection_depth,
        )

        # initial temp from BioViL paper
        # learnable temp from CLIP paper
        self.temperature = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.loss_combo = config.loss_combo

    def contrastive_loss(
        self,
        *,  # enforce kwargs
        img_projs: torch.Tensor,
        txt_projs: torch.Tensor,
        txt_local_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        inputs must be unit vectors of equal dimensionality
        computes symmetric contrastive loss
        """
        if txt_local_mask is not None:
            assert txt_projs.dim() == 3, "Can only use txt_local_mask with local text projections"  # fmt: skip
            assert txt_local_mask.dim() == 2

        # averaging of local embeddings will result in non-unit vectors
        # however, simplifies cross entropy computation and intermediate similarity
        # matrix can be thought of as average of pairwise cosine similarities
        if img_projs.dim() == 3:
            img_projs = img_projs.mean(axis=1)
        if txt_projs.dim() == 3:
            if txt_local_mask is not None:
                # txt_projs  # (B, L, H)
                txt_local_mask = txt_local_mask.unsqueeze(-1)  # (B, L, 1)
                txt_projs = (txt_projs * txt_local_mask).sum(1)  # (B, H)
                txt_projs = txt_projs / txt_local_mask.sum(1)
            else:
                txt_projs = txt_projs.mean(axis=1)

        # temp scaling in form of BioViL paper (divide)
        similarities = img_projs @ txt_projs.T / self.temperature

        # symmetric cross entropy loss
        targets = torch.arange(img_projs.shape[0]).to(img_projs.device)
        loss = (
            F.cross_entropy(similarities, targets)
            + F.cross_entropy(similarities.T, targets)
        ) / 2
        return loss

    def self_repulsive_loss(
        self,
        x: torch.Tensor,
        *,  # enforce kwargs
        local_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        inputs must be unit vectors
        """
        assert x.dim() == 3, "Must do repulsive loss with local embeddings"
        assert local_mask is None or local_mask.dim() == 2
        sims = x @ x.mT  # (B, L, H) @ (B, H, L) --> (B, L, L)

        # minimize similarity between pairs from the same sample,
        # but ignore self pairs i.e. x_i,j @ x_i,j.T
        self_pairs = torch.eye(sims.shape[1], device=sims.device).expand(sims.shape)
        non_self_pairs = 1 - self_pairs

        # ignore masked-out chunks if local mask is provided
        mask = non_self_pairs
        if local_mask is not None:
            local_mask = local_mask.unsqueeze(-1)
            local_mask = local_mask @ local_mask.mT
            mask = mask * local_mask

        masked_sims = (mask * sims).sum()
        return masked_sims / mask.sum()

    def forward(
        self,
        *,  # enforce kwargs
        t_global: torch.Tensor,  # (B, H_t), global text input
        i_global: torch.Tensor,  # (B, H_i), global image input
        t_locals: torch.Tensor,  # (B, L_t, H_t) local text input
        i_locals: torch.Tensor,  # (B, L_i, H_i) local image input
        t_local_mask: torch.Tensor,  # (B, L_t) local text mask
    ) -> torch.Tensor:
        assert t_global.dim() == 2, "Global text embedding is not 2 dimensional"
        assert i_global.dim() == 2, "Global image embedding is not 2 dimensional"
        assert t_locals.dim() == 3, "Local text embedding is not 3 dimensional"
        assert i_locals.dim() == 3, "Local image embedding is not 3 dimensional"  # image locals not flattened before contraster? # fmt: skip
        assert t_local_mask.dim() == 2

        t_global = self.dropout(t_global)
        i_global = self.dropout(i_global)
        t_locals = self.dropout(t_locals)
        i_locals = self.dropout(i_locals)

        t_global = self.proj_t(t_global)
        i_global = self.proj_i(i_global)
        t_locals = self.proj_t(t_locals)
        i_locals = self.proj_i(i_locals)

        t_global = F.normalize(t_global, dim=-1)
        i_global = F.normalize(i_global, dim=-1)
        t_locals = F.normalize(t_locals, dim=-1)
        i_locals = F.normalize(i_locals, dim=-1)

        if self.loss_combo == "ig_tg":
            loss = self.contrastive_loss(img_projs=i_global, txt_projs=t_global)
        elif self.loss_combo == "igl_tg":
            loss = (
                self.contrastive_loss(img_projs=i_global, txt_projs=t_global)
                + self.contrastive_loss(img_projs=i_locals, txt_projs=t_global)
            ) / 2 + self.self_repulsive_loss(i_locals)
        elif self.loss_combo == "ig_tgl":
            loss = (
                self.contrastive_loss(img_projs=i_global, txt_projs=t_global)
                + self.contrastive_loss(img_projs=i_global, txt_projs=t_locals)
            ) / 2 + self.self_repulsive_loss(t_locals)
        elif self.loss_combo == "igl_tgl":
            loss = (
                self.contrastive_loss(img_projs=i_global, txt_projs=t_global)
                + self.contrastive_loss(img_projs=i_global, txt_projs=t_locals)
                + self.contrastive_loss(img_projs=i_locals, txt_projs=t_global)
                + self.contrastive_loss(img_projs=i_locals, txt_projs=t_locals)
            ) / 4 + (
                self.self_repulsive_loss(i_locals) + self.self_repulsive_loss(t_locals)
            ) / 2
        else:
            raise ValueError(f"Unknown loss_combo ({self.loss_combo})")

        return loss
