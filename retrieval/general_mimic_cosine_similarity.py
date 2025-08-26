import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from remix.models import InferenceEngine

BASE_MODEL_PATH = "microsoft/BiomedVLP-CXR-BERT-specialized"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE, file=sys.stderr)


def _get_inference_engine(model_checkpoint_path, tokenizer_path) -> InferenceEngine:
    """
    Creates and returns an instance of the InferenceEngine
    """
    inference_engine = InferenceEngine(model_checkpoint_path, tokenizer_path)
    return inference_engine


def np_array_to_torch_tensor(np_array, device):
    """
    Takes a numpy array and converts it to a torch tensor with datatype float32
    """
    torch_tensor = torch.tensor(np_array, dtype=torch.float32, device=device)
    return torch_tensor


def img_embeddings_df(path, split_type):
    """
    Takes a file path for a CSV that holds information about image embeddings
    Filters the dataframe based on the split type
    Returns a Pandas dataframe
    """
    embeddings_pd = pd.read_csv(path)
    embeddings_pd = embeddings_pd[embeddings_pd["split"] == split_type]
    return embeddings_pd


def create_pos_search_queries_for_label(label):
    """
    Takes a label (string) and creates a list of positive search queries for the given label
    """
    pos_search_queries = [
        f"Findings consistent with {label}",
        f"Findings suggesting {label}",
        f"Findings are most compatible with {label}",
        f"{label} seen",
    ]
    return pos_search_queries


def create_search_queries_embedding(search_queries, inference_engine, device=DEVICE):
    """
    Takes a list of search queries and returns a list of embeddings for each search query
    """
    search_queries_embeddings = []
    for query in search_queries:
        embedding = inference_engine.get_projected_text_embedding(query).to(device)
        search_queries_embeddings.append(embedding)
    return search_queries_embeddings


def find_cosine_similarity(img_embedding, search_query_embedding):
    """
    Takes an image embedding and a search query embedding as torch tensors and returns the cosine
    similarity score
    Image embedding size should be ([128]) and text embedding is ([1, 128])
    Both inputs will be reshaped to ensure consistent 2D matrix multiplication
    """
    # Ensure both are 2D for consistent matrix operations
    img_embedding_reshaped = img_embedding.reshape(1, 128)  # [128] -> [1, 128]

    # search_query_embedding is always [1, 128] from our fixed text embedding method
    text_embedding = F.normalize(
        search_query_embedding, dim=1, p=2
    )  # Normalize along feature dimension

    # Simple matrix multiplication: [1, 128] @ [128, 1] = [1, 1]
    cos_similarity = img_embedding_reshaped @ text_embedding.t()
    return cos_similarity.item()  # Convert to scalar


def convert_string_to_np(embedding_str):
    """
    Converts a string to a numpy array
    """
    return np.fromstring(embedding_str[1:-1], sep=" ")


def load_queries_from_csv(csv_path):
    """
    Loads queries from a CSV file with columns 'Variable' and 'Text'
    Returns a dictionary mapping labels to lists of queries
    """
    queries_df = pd.read_csv(csv_path)
    queries_dict = {}
    for label in queries_df["Variable"].unique():
        queries_dict[label] = queries_df[queries_df["Variable"] == label][
            "Text"
        ].tolist()
    return queries_dict


def main(
    model_checkpoint_path,
    embedding_library_path,
    output_file_path,
    split_type,
    label_type,
    labeled,
):
    checkpoint_inference_engine = _get_inference_engine(
        model_checkpoint_path, BASE_MODEL_PATH
    )

    # Use the hardcoded convirt queries path from the original file
    # External path, do not change
    CONVIRT_QUERIES_CSV = (
        "/opt/gpudata/imadejski/search-model/remix/data/convirt_queries.csv"
    )

    # Determine which labels and queries to use
    if label_type == "convirt":
        queries_dict = load_queries_from_csv(CONVIRT_QUERIES_CSV)
        labels = list(queries_dict.keys())
    elif label_type == "raw":
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
    elif label_type == "auto":
        labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "Airspace Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]
    else:
        raise ValueError("label_type must be one of: convirt, raw, auto")

    embeddings_library_pd = img_embeddings_df(embedding_library_path, split_type)

    # If labeled, filter the dataframe to only include labeled rows
    if labeled:
        if "labeled" not in embeddings_library_pd.columns:
            raise ValueError(
                "The embedding library does not have a 'labeled' column for filtering."
            )
        embeddings_library_pd = embeddings_library_pd[
            embeddings_library_pd["labeled"] == 1
        ]

    for label in labels:
        # Determine queries for each label_type
        if label_type == "convirt":
            search_queries = queries_dict[label]
        elif label_type == "auto":
            search_queries = create_pos_search_queries_for_label(label)
        elif label_type == "raw":
            search_queries = [label]
        else:
            raise ValueError("label_type must be one of: convirt, raw, auto")

        pos_embeddings = create_search_queries_embedding(
            search_queries, checkpoint_inference_engine, DEVICE
        )

        # Output columns depend on label_type
        if label_type in ["convirt", "auto"]:
            for i in range(len(search_queries)):
                embeddings_library_pd[f"{label} cosine_similarity_{i+1}"] = np.nan
            embeddings_library_pd[f"{label} average_cosine_similarity"] = np.nan
            embeddings_library_pd[f"{label} max_cosine_similarity"] = np.nan
        elif label_type == "raw":
            embeddings_library_pd[f"{label} cosine_similarity"] = np.nan

        for index, row in embeddings_library_pd.iterrows():
            img_embedding_string = row["embedding"]
            img_embedding_np = convert_string_to_np(img_embedding_string)
            img_embedding = np_array_to_torch_tensor(img_embedding_np, DEVICE)

            if label_type in ["convirt", "auto"]:
                cosine_similarities = []
                for i, pos_embedding in enumerate(pos_embeddings):
                    cosine_similarity = find_cosine_similarity(
                        img_embedding, pos_embedding
                    )
                    embeddings_library_pd.at[
                        index, f"{label} cosine_similarity_{i+1}"
                    ] = cosine_similarity
                    cosine_similarities.append(cosine_similarity)
                if cosine_similarities:
                    average_of_cosine_similarities = np.mean(cosine_similarities)
                    embeddings_library_pd.at[
                        index, f"{label} average_cosine_similarity"
                    ] = average_of_cosine_similarities

                    max_of_cosine_similarities = np.max(cosine_similarities)
                    embeddings_library_pd.at[
                        index, f"{label} max_cosine_similarity"
                    ] = max_of_cosine_similarities
            elif label_type == "raw":
                cosine_similarity = find_cosine_similarity(
                    img_embedding, pos_embeddings[0]
                )
                embeddings_library_pd.at[index, f"{label} cosine_similarity"] = (
                    cosine_similarity
                )

    embeddings_library_pd.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generalized cosine similarity calculation for MIMIC embeddings"
    )
    parser.add_argument(
        "model_checkpoint_path",
        type=str,
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "embedding_library_path",
        type=str,
        help="Path to the embedding library CSV file",
    )
    parser.add_argument(
        "output_file_path", type=str, help="Path to save the output CSV file"
    )
    parser.add_argument(
        "split_type", type=str, help="Type of data split (e.g., validate, train, test)"
    )
    parser.add_argument(
        "--label_type",
        type=str,
        choices=["convirt", "raw", "auto"],
        default="auto",
        help="Which label/query mode to use: convirt (CSV), raw (label only), or auto (expanded queries)",
    )
    parser.add_argument(
        "--labeled",
        action="store_true",
        help="If set, only use rows with labeled==1 in the embedding library",
    )

    args = parser.parse_args()
    main(
        args.model_checkpoint_path,
        args.embedding_library_path,
        args.output_file_path,
        args.split_type,
        args.label_type,
        args.labeled,
    )
