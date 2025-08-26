import argparse
import os
from typing import Literal, get_args

import numpy as np
import pandas as pd
import torch
from gloria_inference_engine import GLoRIAInferenceEngine
from health_multimodal.image import get_image_inference
from tqdm import tqdm

from remix.datasets import get_image_paths, get_merged_df
from remix.models import InferenceEngine, InferenceEngineV2

MODEL_T = Literal["biovil", "gloria", "remix-v1", "remix-v2"]


def main(
    *,  # enforce kwargs
    model_type: MODEL_T,
    model_checkpoint: str | None = None,
    tokenizer_checkpoint: str | None = None,
    splits_path: str,
    metadata_path: str,
    img_dir: str,
    img_ext: str,
    split: str,
    frontal_only: bool,
    one_image_per_study: bool,
    output_dir: str,
    variant: str | None = None,
):
    info_path = os.path.join(output_dir, f"{split}-image-info.csv")
    if variant is None:
        variant = ""
    else:
        variant = f"{variant}-"
    emb_path = os.path.join(
        output_dir,
        f"{split}-{model_type}-{variant}image-embeddings.npy",
    )
    os.makedirs(output_dir, exist_ok=True)
    assert not os.path.exists(emb_path), f"{emb_path} already exists, please remove before proceeding" # fmt: skip

    if model_type == "biovil":
        assert model_checkpoint is None, "Do not provide model_checkpoint if using model_type=biovil"  # fmt: skip
        assert tokenizer_checkpoint is None, "Do not provide tokenizer_checkpoint if using model_type=biovil"  # fmt: skip
        inferencer = get_image_inference("biovil")
    elif model_type == "gloria":
        assert model_checkpoint is not None, "Must provide model_checkpoint if using model_type=gloria"  # fmt: skip
        assert tokenizer_checkpoint is None, "Do not provide tokenizer_checkpoint if using model_type=gloria"  # fmt: skip
        inferencer = GLoRIAInferenceEngine(model_checkpoint)
    elif model_type == "v1":
        assert model_checkpoint is not None and tokenizer_checkpoint is not None, "Must provide both model_checkpoint and tokenizer_checkpoint if using model_type=v1"  # fmt: skip
        inferencer = InferenceEngine(model_checkpoint, tokenizer_checkpoint)
    elif model_type == "v2":
        assert model_checkpoint is not None, "Must provide model_checkpoint if using model_type=v2"  # fmt: skip
        assert tokenizer_checkpoint is None, "Do not provide tokenizer_checkpoint if using model_type=v2"  # fmt: skip
        inferencer = InferenceEngineV2(model_checkpoint)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    splits_df = pd.read_csv(splits_path)
    metadata_df = pd.read_csv(metadata_path)
    df = get_merged_df(
        splits_df=splits_df,
        metadata_df=metadata_df,
        split=split,
        frontal_only=frontal_only,
        one_image_per_study=one_image_per_study,
    )
    image_paths = get_image_paths(
        split=split,
        img_dir=img_dir,
        img_ext=img_ext,
        subject_ids=df["subject_id"],
        study_ids=df["study_id"],
        dicom_ids=df["dicom_id"],
    )
    projs = []
    for image_path in tqdm(image_paths):
        if model_type == "biovil":
            proj = inferencer.get_projected_global_embedding(image_path)
        elif model_type == "gloria":
            proj = inferencer.get_image_projection(image_path=image_path)
        elif model_type == "v1":
            proj = inferencer.get_projected_global_embeddings(image_path)
        elif model_type == "v2":
            proj = inferencer.get_image_projection(image_path=image_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if isinstance(proj, torch.Tensor):
            proj = proj.detach().cpu().numpy()
        assert isinstance(proj, np.ndarray)
        assert proj.ndim == 1
        projs.append(proj)
    projs = np.stack(projs)

    np.save(emb_path, projs)
    if not os.path.exists(info_path):
        df.to_csv(info_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=get_args(MODEL_T))
    parser.add_argument("--model_checkpoint")
    parser.add_argument("--tokenizer_checkpoint")
    parser.add_argument("--splits_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--img_ext", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--frontal_only", type=bool, default=True)
    parser.add_argument("--one_image_per_study", type=bool, default=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant")
    args = parser.parse_args()

    main(
        model_type=args.model_type,
        model_checkpoint=args.model_checkpoint,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        splits_path=args.splits_path,
        metadata_path=args.metadata_path,
        img_dir=args.img_dir,
        img_ext=args.img_ext,
        split=args.split,
        frontal_only=args.frontal_only,
        one_image_per_study=args.one_image_per_study,
        output_dir=args.output_dir,
        variant=args.variant,
    )
