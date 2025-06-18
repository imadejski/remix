import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gloria.builder import build_text_model
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", choices=["base", "mimic"], required=True)
    parser.add_argument("--split", choices=["train", "validate", "test"], required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--frontal_impression_only",
        action="store_true",
        help="Only process images that have frontal impressions",
    )
    args = parser.parse_args()
    if args.weights == "base":
        args.weights_path = "pretrained/chexpert_resnet50.ckpt"
    elif args.weights == "mimic":
        args.weights_path = (
            "data/ckpt/gloria_pretrain_1.0/2025_01_10_10_55_38/epoch=9-step=41549.ckpt"
        )
    return args


def create_pos_search_queries_for_label(label):
    return [
        f"Findings consistent with {label}",
        f"Findings suggesting {label}",
        f"Findings are most compatible with {label}",
        f"{label} seen",
    ]


def main(args):
    # Load config and setup model
    cfg = OmegaConf.load("configs/chexpert_pretrain_config.yaml")
    cfg.data.text.full_report = True

    text_encoder = build_text_model(cfg)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.bert_type)

    # Load weights
    sd = torch.load(args.weights_path, "cpu")["state_dict"]
    text_sd = {
        k.replace("gloria.text_encoder.", ""): v
        for k, v in sd.items()
        if k.startswith("gloria.text_encoder.")
    }
    text_encoder.load_state_dict(text_sd)
    text_encoder = text_encoder.eval().cuda()

    # Load image embeddings and metadata
    img_embs = np.load(
        os.path.join(
            args.input_dir, f"gloria-{args.weights}-{args.split}-split-img-embeds.npy"
        )
    )
    metadata_df = pd.read_csv(
        os.path.join(
            args.input_dir, f"gloria-{args.weights}-{args.split}-split-metadata.csv"
        )
    )

    # Filter for frontal impression only if flag is set
    if args.frontal_impression_only:
        frontal_df = pd.read_csv(
            "/opt/gpudata/imadejski/mimic-cxr/mimic-cxr-2.0.0-frontal-impression-only.csv"
        )
        metadata_df = metadata_df[metadata_df["dicom_ids"].isin(frontal_df["dicom_id"])]
        img_embs = img_embs[metadata_df.index]

    # Convert image embeddings to torch
    img_embs = torch.tensor(img_embs).cuda()

    # Define labels
    labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    # Process each label
    for label in tqdm(labels, desc="Processing labels"):
        queries = create_pos_search_queries_for_label(label)

        # Get text embeddings for all queries
        query_embs = []
        for query in queries:
            # Tokenize and encode text
            tokens = tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=cfg.data.text.word_num,
            )

            with torch.no_grad():
                _, text_emb, _ = text_encoder(
                    tokens["input_ids"].cuda(),
                    tokens["attention_mask"].cuda(),
                    tokens["token_type_ids"].cuda(),
                )
            query_embs.append(text_emb)

        # Calculate cosine similarities
        query_embs = torch.cat(query_embs, dim=0)

        # Add individual cosine similarities
        for i in range(len(queries)):
            cos_sim = F.cosine_similarity(img_embs, query_embs[i : i + 1], dim=1)
            metadata_df[f"{label} cosine_similarity_{i+1}"] = cos_sim.cpu().numpy()

        # Calculate and add average and max similarities
        all_sims = torch.stack(
            [
                F.cosine_similarity(img_embs, query_emb.unsqueeze(0), dim=1)
                for query_emb in query_embs
            ]
        )

        metadata_df[f"{label} average_cosine_similarity"] = (
            all_sims.mean(dim=0).cpu().numpy()
        )
        metadata_df[f"{label} max_cosine_similarity"] = (
            all_sims.max(dim=0)[0].cpu().numpy()
        )

    # Save results
    metadata_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
