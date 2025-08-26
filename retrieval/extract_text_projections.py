import argparse
import os
from typing import Literal, get_args

import numpy as np
import pandas as pd
import torch
from gloria_inference_engine import GLoRIAInferenceEngine
from health_multimodal.text import get_bert_inference
from tqdm import tqdm

from remix.models import InferenceEngine, InferenceEngineV2

MODEL_T = Literal["biovil", "gloria", "remix-v1", "remix-v2"]


def main(
    *,  # enforce kwargs
    model_type: MODEL_T,
    model_checkpoint: str | None = None,
    tokenizer_checkpoint: str | None = None,
    query_csv: str,
    output_dir: str,
    variant: str | None = None,
):
    if variant is None:
        variant = ""
    else:
        variant = f"{variant}-"
    query_name = os.path.splitext(os.path.basename(query_csv))[0]
    emb_path = os.path.join(
        output_dir,
        f"{query_name}-{model_type}-{variant}text-embeddings.npy",
    )
    os.makedirs(output_dir, exist_ok=True)
    assert not os.path.exists(emb_path), f"{emb_path} already exists, please remove before proceeding" # fmt: skip

    if model_type == "biovil":
        assert model_checkpoint is None, "Do not provide model_checkpoint if using model_type=biovil"  # fmt: skip
        assert tokenizer_checkpoint is None, "Do not provide tokenizer_checkpoint if using model_type=biovil"  # fmt: skip
        inferencer = get_bert_inference("cxr_bert")
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

    queries = pd.read_csv(query_csv)["Text"]
    projs = []
    for query in tqdm(queries):
        if model_type == "biovil":
            proj = inferencer.get_embeddings_from_prompt(query)
        elif model_type == "gloria":
            proj = inferencer.get_text_projection(text=query)
        elif model_type == "v1":
            proj = inferencer.get_projected_text_embedding(query)
        elif model_type == "v2":
            proj = inferencer.get_text_projection(text=query)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if isinstance(proj, torch.Tensor):
            proj = proj.detach().cpu().numpy()
        assert isinstance(proj, np.ndarray)
        assert proj.ndim == 1
        projs.append(proj)
    projs = np.stack(projs)

    np.save(emb_path, projs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=get_args(MODEL_T))
    parser.add_argument("--model_checkpoint")
    parser.add_argument("--tokenizer_checkpoint")
    parser.add_argument("--query_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant")
    args = parser.parse_args()

    main(
        model_type=args.model_type,
        model_checkpoint=args.model_checkpoint,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        query_csv=args.query_csv,
        output_dir=args.output_dir,
        variant=args.variant,
    )
