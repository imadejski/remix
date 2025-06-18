from typing import Literal

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

from .utils import ResNet50Transform  # isort: skip

nltk.download("punkt")


SECTION_T = Literal["findings", "impression"]
SPLIT_T = Literal["train", "validate", "test"]

RANKED_VIEWS = [
    "PA",
    "AP",
    "LATERAL",
    "LL",
    "AP AXIAL",
    "AP LLD",
    "AP RLD",
    "PA RLD",
    "PA LLD",
    "LAO",
    "RAO",
    "LPO",
    "XTABLE LATERAL",
    "SWIMMERS",
    "",
]

FRONTAL_VIEWS = ["PA", "AP"]


class CXRDataset(Dataset):
    def __init__(
        self,
        *,
        img_dir: str,
        img_ext: str,
        splits_df: pd.DataFrame,
        split: SPLIT_T,
        notes_df: pd.DataFrame,
        section: SECTION_T,
        metadata_df: pd.DataFrame,
        frontal_only: bool,
        one_image_per_study: bool,
        num_chunks: int,
        num_overlap: int,
        text_tokenizer: PreTrainedTokenizer,
        mlm_probability: float = 0.0,
    ):
        assert text_tokenizer.padding_side == "right"
        assert text_tokenizer.pad_token is not None
        assert text_tokenizer.model_max_length is not None
        assert num_chunks > 0
        assert notes_df["study_id"].is_unique
        assert splits_df["dicom_id"].is_unique
        assert metadata_df["dicom_id"].is_unique

        # fmt: off
        notes_df = notes_df[
            (
                (notes_df[section].notna()) &
                (notes_df[section].str.strip() != "")
            )
        ]
        # fmt: on
        splits_df = splits_df[splits_df["split"] == split]
        views = FRONTAL_VIEWS if frontal_only else RANKED_VIEWS
        view_order = {v: i for i, v in enumerate(views)}
        metadata_df = metadata_df[metadata_df["ViewPosition"].isin(view_order)].copy()
        metadata_df["ViewPosition"] = metadata_df["ViewPosition"].replace(view_order)
        metadata_df = metadata_df.sort_values(["study_id", "ViewPosition", "dicom_id"])
        if one_image_per_study:
            metadata_df = metadata_df.drop_duplicates("study_id", keep="first")
        df = (
            metadata_df[["subject_id", "study_id", "dicom_id"]]
            .merge(notes_df[["study_id", section]], on="study_id")
            .merge(splits_df[["dicom_id", "study_id"]])
            .sort_values(["subject_id", "study_id", "dicom_id"])
        )

        self.image_paths = (
            img_dir
            + "/p"
            + df["subject_id"].astype(str).str[:2]
            + "/p"
            + df["subject_id"].astype(str)
            + "/s"
            + df["study_id"].astype(str)
            + "/"
            + df["dicom_id"]
            + img_ext
        )
        self.transform = ResNet50Transform()

        self.notes = df[section]
        self.num_chunks = num_chunks
        self.num_overlap = num_overlap
        self.text_tokenizer = text_tokenizer
        self.masker = None
        if mlm_probability > 0:
            self.masker = DataCollatorForLanguageModeling(
                tokenizer=text_tokenizer,
                mlm=True,
                mlm_probability=mlm_probability,
            )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        note = self.notes[index]
        chunks = get_chunked_note(
            note=note,
            num_chunks=self.num_chunks,
            num_overlap=self.num_overlap,
        )
        im_path = self.image_paths[index]

        note_tok = self.text_tokenizer(
            note,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )  # (1, T)

        chunks_tok = self.text_tokenizer(
            chunks,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )  # (Lx, T)

        im = read_image(im_path)
        assert im.shape[0] == 1  # greyscale
        im = Image.fromarray(im.numpy()[0]).convert("L")
        im = self.transform(im)

        input_ids_global = note_tok["input_ids"]
        input_ids_locals = chunks_tok["input_ids"]

        if self.masker is not None:
            input_ids_global, labels_global = self.masker.torch_mask_tokens(
                input_ids_global,
            )
            input_ids_locals, labels_locals = self.masker.torch_mask_tokens(
                input_ids_locals,
            )

        ret = {
            "input_ids_global": input_ids_global[0],
            "attention_mask_global": note_tok["attention_mask"][0],
            "input_ids_locals": input_ids_locals,
            "attention_mask_locals": chunks_tok["attention_mask"],
            "images": im,
        }

        if self.masker is not None:
            ret["labels_global"] = labels_global[0]
            ret["labels_locals"] = labels_locals

        return ret

    def __len__(self):
        return len(self.image_paths)


def get_chunked_note(
    *,  # enforce kwargs
    note: str,
    num_chunks: int,
    num_overlap: int = 0,
) -> list[str]:
    if num_overlap != 0:
        raise NotImplemented("Overlapping chunks not yet supported")
    sents = sent_tokenize(note)
    n_sents = len(sents)
    if n_sents < num_chunks:
        # fill remainder with full report
        num_fill = num_chunks - n_sents
        chunks = sents + [note] * num_fill
    else:
        # join sentences into specified number of chunks
        chunks = []
        chunk_size = n_sents // num_chunks
        remainder = n_sents % num_chunks
        bins = [chunk_size + (1 if i < remainder else 0) for i in range(num_chunks)]
        edges = [0] + list(np.cumsum(bins))
        for lo, hi in zip(edges[:-1], edges[1:]):
            chunk = " ".join(sents[lo:hi])
            chunks.append(chunk)
    return chunks
