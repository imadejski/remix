import argparse
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from remix.models import InferenceEngine

RESIZE = 512
CENTER_CROP_SIZE = 512

BASE_MODEL_PATH = "microsoft/BiomedVLP-CXR-BERT-specialized"

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE, file=sys.stderr)


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


def _get_image_inference_engine(model_checkpoint_path):
    """
    Defines image inference model from fine-tuned model.
    Includes resizing and cropping to image.
    """
    image_inference = InferenceEngine(model_checkpoint_path, BASE_MODEL_PATH)
    return image_inference


def convert_tensor_to_np_array(tensor):
    """
    Returns numpy array from Torch tensor.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    numpy_array = tensor.detach().numpy()
    return numpy_array


def get_image_embedding(image_path, inference_engine):
    """
    Returns numpy array of l2-normalized global image embedding from
    image inference model
    """

    image_embedding = inference_engine.get_projected_global_embeddings(image_path)
    np_img_embedding = convert_tensor_to_np_array(image_embedding)
    return np_img_embedding


def main(
    model_checkpoint_path,
    embedding_library_output_path,
    split_type,
    frontal_impression_only,
    split_file_path,
):
    image_inference = _get_image_inference_engine(model_checkpoint_path)

    if split_file_path:
        split_file = split_file_path
    elif frontal_impression_only:
        split_file = "/opt/gpudata/imadejski/mimic-cxr/mimic-cxr-2.0.0-frontal-impression-only.csv"
    else:
        split_file = "/opt/gpudata/mimic-cxr/mimic-cxr-2.0.0-split.csv"

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
    parser = argparse.ArgumentParser(description="Get paths")
    parser.add_argument(
        "model_checkpoint_path",
        type=str,
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the output CSV file"
    )
    parser.add_argument(
        "split_type", type=str, help="Type of data split (e.g., validate, train, test)"
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
        args.model_checkpoint_path,
        args.output_path,
        args.split_type,
        args.frontal_impression_only,
        args.split_file_path,
    )
