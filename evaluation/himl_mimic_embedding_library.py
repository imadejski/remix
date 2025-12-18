import argparse
import os
import sys
from enum import Enum, unique
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model.pretrained import (
    get_biovil_image_encoder,
    get_biovil_t_image_encoder,
)
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine
from torchvision.datasets.utils import check_integrity
from tqdm import tqdm

RESIZE = 512
CENTER_CROP_SIZE = 512

device_index = os.getenv("CUDA_VISIBLE_DEVICES", "1")
device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", file=sys.stderr)


def create_split_df(file_path, split_type):
    """Creates the dataframe with the specified split type data and initializes a file_path column"""
    split_df = pd.read_csv(file_path)
    split_type_df = split_df[split_df["split"].isin([split_type])]
    return split_type_df


def create_paths(df):
    """Creates a column with the path for each individual image,
    takes a dataframe with columns subject_ids, study_ids, and dicom_ids"""
    for index, row in df.iterrows():
        # retrieve each sub folder value to get path
        patient_id = row["subject_id"]
        study_id = row["study_id"]
        dicom_id = row["dicom_id"]

        # assign path in df
        path = (
            "/opt/gpudata/mimic-cxr/files/p"
            + str(patient_id)[:2]
            + "/p"
            + str(patient_id)
            + "/s"
            + str(study_id)
            + "/"
            + str(dicom_id)
            + ".jpg"
        )
        df.loc[index, "file_path"] = path
        df.loc[index, "study_id"] = study_id
    return df


def _get_image_inference_engine(model_type="biovil"):
    """
    Defines image inference model from BioVIL or BioVIL-T image encoder.
    Applies resizing and cropping to image.

    Args:
        model_type: Either "biovil" or "biovil-t"
    """
    if model_type.lower() == "biovil-t":
        image_model = get_biovil_t_image_encoder().to(device)
    elif model_type.lower() == "biovil":
        image_model = get_biovil_image_encoder().to(device)
    else:
        raise ValueError(
            f"Invalid model type: {model_type}. Must be 'biovil' or 'biovil-t'"
        )

    image_inference = ImageInferenceEngine(
        image_model=image_model,
        transform=create_chest_xray_transform_for_inference(
            resize=RESIZE, center_crop_size=CENTER_CROP_SIZE
        ),
    )
    return image_inference


def convert_tensor_to_np_array(tensor):
    """
    Returns numpy array from Torch tensor.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    numpy_array = tensor.numpy()
    return numpy_array


def get_image_embedding(image_path, inference_engine):
    """
    Returns numpy array of l2-normalized global image embedding from
    image inference model
    """

    image_embedding = inference_engine.get_projected_global_embedding(
        image_path=Path(image_path)
    )
    np_img_embedding = convert_tensor_to_np_array(image_embedding)
    return np_img_embedding


def main(
    embedding_library_output_path,
    split_type,
    model_type,
    frontal_impression_only,
    split_file_path,
):
    image_inference = _get_image_inference_engine(model_type)

    if split_file_path:
        split_file = split_file_path
    elif frontal_impression_only:
        split_file = "/opt/gpudata/imadejski/mimic-cxr/mimic-cxr-2.0.0-frontal-impression-only.csv"
    else:
        split_file = "/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv.gz"

    split_df = create_split_df(split_file, split_type)
    split_df = create_paths(split_df)

    split_df["embedding"] = np.nan

    tqdm.pandas(desc="Calculating Embeddings")
    split_df["embedding"] = split_df["file_path"].progress_apply(
        get_image_embedding, inference_engine=image_inference
    )

    split_df.to_csv(embedding_library_output_path, index=False)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate image embeddings using BioVIL or BioVIL-T models"
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the output CSV file"
    )
    parser.add_argument(
        "split_type", type=str, help="Type of data split (e.g., validate, train, test)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["biovil", "biovil-t"],
        default="biovil",
        help="Type of model to use: 'biovil' or 'biovil-t' (default: biovil)",
    )
    parser.add_argument(
        "--frontal_impression_only",
        action="store_true",
        help="Use the frontal impression only CSV file (ignored if --split_file_path is set)",
    )
    parser.add_argument(
        "--split_file_path",
        type=str,
        default=None,
        help="Custom path to the split CSV file. If not set, uses default or frontal-impression-only based on flag.",
    )

    args = parser.parse_args()
    main(
        args.output_path,
        args.split_type,
        args.model_type,
        args.frontal_impression_only,
        args.split_file_path,
    )
