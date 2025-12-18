import argparse
from typing import Sequence

import numpy as np
import pandas as pd

FRONTAL_VIEWS = ["PA", "AP"]


def select_frontal_dicom_per_study(metadata_df: pd.DataFrame) -> pd.DataFrame:
    metadata_df = metadata_df[metadata_df["ViewPosition"].isin(FRONTAL_VIEWS)].copy()
    view_rank = {"PA": 0, "AP": 1}
    metadata_df["view_rank"] = metadata_df["ViewPosition"].map(view_rank)
    metadata_df = metadata_df.sort_values(["study_id", "view_rank", "dicom_id"])
    dedup = metadata_df.drop_duplicates("study_id", keep="first")
    return dedup[["subject_id", "study_id", "dicom_id"]]


def build_exclusive_positive_mask(
    labels_df: pd.DataFrame, labels: Sequence[str]
) -> pd.Series:
    for label in labels:
        if label not in labels_df.columns:
            raise ValueError(f"Label column '{label}' not found in CheXpert CSV")

    pos_mask = labels_df[list(labels)].eq(1.0)
    num_pos = pos_mask.sum(axis=1)
    exclusive_any = num_pos == 1
    return exclusive_any


def sample_per_label(
    labels_df: pd.DataFrame,
    frontal_one_per_study_df: pd.DataFrame,
    labels: Sequence[str],
    num_per_label: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    merged = labels_df.merge(
        frontal_one_per_study_df, on=["subject_id", "study_id"], how="inner"
    )

    for label in labels:
        pos_mask = merged[label].eq(1.0)
        other_labels = [l for l in labels if l != label]
        not_pos_others = ~merged[other_labels].eq(1.0).any(axis=1)
        subset = merged[pos_mask & not_pos_others]
        if subset.empty:
            raise RuntimeError(
                f"No candidates found for label '{label}' with exclusivity condition"
            )

        if len(subset) < num_per_label:
            sampled = subset
        else:
            indices = rng.choice(subset.index.values, size=num_per_label, replace=False)
            sampled = subset.loc[indices]

        sampled = sampled[["subject_id", "study_id", "dicom_id"]].copy()
        sampled["split"] = "custom"
        sampled["exclusive_label"] = label
        rows.append(sampled)

    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate frontal exclusive-positive MIMIC-CXR subset (N per label)"
    )
    parser.add_argument(
        "--chexpert_csv",
        type=str,
        default="/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv",
        help="Path to CheXpert label CSV (per study)",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-metadata.csv",
        help="Path to MIMIC metadata CSV (per image)",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Where to write the split CSV"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="Atelectasis,Cardiomegaly,Edema,Pleural Effusion",
        help="Comma-separated labels to include",
    )
    parser.add_argument(
        "--num_per_label", type=int, default=200, help="Number of images per label"
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed for sampling")
    parser.add_argument(
        "--split_csv",
        type=str,
        default="/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv",
        help="Path to the MIMIC split CSV (subject_id,study_id,dicom_id,split)",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="test",
        help="Split type to restrict candidates to (e.g., test)",
    )

    args = parser.parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    chexpert_df = pd.read_csv(args.chexpert_csv)
    # ensure only one row per study_id in labels df
    chexpert_df = chexpert_df.drop_duplicates(["subject_id", "study_id"], keep="first")

    metadata_df = pd.read_csv(
        args.metadata_csv,
        usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"],
    )
    frontal_one = select_frontal_dicom_per_study(metadata_df)

    # Restrict to specified split
    split_df = pd.read_csv(
        args.split_csv, usecols=["subject_id", "study_id", "dicom_id", "split"]
    )
    split_df = split_df[split_df["split"] == args.split_type]
    allowed_studies = set(split_df["study_id"].unique().tolist())
    chexpert_df = chexpert_df[chexpert_df["study_id"].isin(allowed_studies)].copy()
    frontal_one = frontal_one[frontal_one["study_id"].isin(allowed_studies)].copy()

    exclusive_mask = build_exclusive_positive_mask(chexpert_df, labels)
    chexpert_exclusive = chexpert_df[exclusive_mask].copy()

    subset = sample_per_label(
        chexpert_exclusive,
        frontal_one,
        labels,
        args.num_per_label,
        args.seed,
    )

    # Set split to requested split type for downstream filtering
    subset["split"] = args.split_type
    subset.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
