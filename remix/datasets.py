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

from .utils import BioViLTransform, GLoRIATransform  # isort: skip

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
        num_chunks: int | None,
        num_overlap: int | None,
        max_chunks: int | None = None,
        text_tokenizer: PreTrainedTokenizer,
        mlm_probability: float = 0.0,
    ):
        assert text_tokenizer.padding_side == "right"
        assert text_tokenizer.pad_token is not None
        assert text_tokenizer.model_max_length is not None
        assert notes_df["study_id"].is_unique
        assert splits_df["dicom_id"].is_unique
        assert metadata_df["dicom_id"].is_unique
        if max_chunks is not None:
            assert num_chunks is None and num_overlap is None
        elif num_chunks is not None or num_overlap is not None:
            assert num_chunks is not None
            assert num_overlap is not None
            assert max_chunks is None
            assert num_chunks > 0
        else:
            raise ValueError("must provide one of max_chunks or num_chunks+num_overlap")

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

        self.image_paths = get_image_paths(
            split=split,
            img_dir=img_dir,
            img_ext=img_ext,
            subject_ids=df["subject_id"],
            study_ids=df["study_id"],
            dicom_ids=df["dicom_id"],
        )

        if "mimic" in img_dir:
            self.transform = BioViLTransform()
        elif "chexpertplus" in img_dir:
            self.transform = GLoRIATransform()
        else:
            raise ValueError(
                f"Cannot infer from image directory path ({img_dir}) "
                f"which transform style to use."
            )

        self.notes = df[section]
        self.num_chunks = num_chunks
        self.num_overlap = num_overlap
        self.max_chunks = max_chunks
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
        if self.num_chunks is not None:
            chunks, chunk_mask = chunk_to_num(
                note=note,
                num_chunks=self.num_chunks,
                num_overlap=self.num_overlap,
            )
        else:
            chunks, chunk_mask = chunk_to_sent(
                note=note,
                pad_truncate=self.max_chunks,
            )

        assert chunk_mask.sum() > 0, f"Sample {index} has no sentences"
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
            # only do MLM on global
            input_ids_global, labels_global = self.masker.torch_mask_tokens(
                input_ids_global,
            )

        ret = {
            "input_ids_global": input_ids_global[0],
            "attention_mask_global": note_tok["attention_mask"][0],
            "input_ids_locals": input_ids_locals,
            "attention_mask_locals": chunks_tok["attention_mask"],
            "local_mask": chunk_mask,  # (Lx,)
            "images": im,
        }

        if self.masker is not None:
            ret["labels_global"] = labels_global[0]

        return ret

    def __len__(self):
        return len(self.image_paths)


def chunk_to_sent(
    *,  # enforce kwargs
    note: str,
    pad_truncate: int,
) -> tuple[list[str], torch.Tensor]:
    chunks = sent_tokenize(note)
    n_sent = len(chunks)
    if n_sent > pad_truncate:
        chunks = chunks[:pad_truncate]
    else:
        chunks = chunks + [""] * (pad_truncate - n_sent)

    chunk_mask = torch.zeros(pad_truncate)
    chunk_mask[:n_sent] = 1
    return chunks, chunk_mask


def chunk_to_num(
    *,  # enforce kwargs
    note: str,
    num_chunks: int,
    num_overlap: int = 0,
) -> tuple[list[str], torch.Tensor]:
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
    return chunks, torch.ones(num_chunks)


def get_mimic_paths(
    *,  # enforce kwargs
    img_dir: str,
    img_ext: str,
    subject_ids: pd.Series,
    study_ids: pd.Series,
    dicom_ids: pd.Series,
) -> pd.Series:
    return (
        img_dir
        + "/p"
        + subject_ids.astype(str).str[:2]
        + "/p"
        + subject_ids.astype(str)
        + "/s"
        + study_ids.astype(str)
        + "/"
        + dicom_ids
        + img_ext
    )


def get_chexpertplus_paths(
    *,  # enforce kwargs
    img_dir: str,
    img_ext: str,
    subject_ids: pd.Series,
    study_ids: pd.Series,
    dicom_ids: pd.Series,
    split: SPLIT_T,
) -> pd.Series:
    split_dir = "train"
    if split == "test":
        # chexpertplus came with only train/valid splits
        # the provided valid split was renamed test
        # a new validate split was created from the train split
        split_dir = "valid"

    # subject, study, and dicom IDs were joined with underscores to create
    # globally unique IDs at each level, however original image data paths
    # were not modified so deconstruct study and dicom IDs to get path
    # fmt: off
    return (
        img_dir
        + "/"
        + split_dir
        + "/"
        + subject_ids
        + "/"
        + study_ids.str.split("_").str[-1]
        + "/"
        + dicom_ids.str.split("_").str[-2]  # this is first part of dicom_id, e.g. view1
        + "_"
        + dicom_ids.str.split("_").str[-1]  # this is second part of dicom_id, e.g. frontal
        + img_ext
    )
    # fmt: on


def get_image_paths(
    *,  # enforce kwargs
    img_dir: str,
    img_ext: str,
    subject_ids: pd.Series,
    study_ids: pd.Series,
    dicom_ids: pd.Series,
    split: SPLIT_T,
) -> pd.Series:
    if "mimic" in img_dir:
        image_paths = get_mimic_paths(
            img_dir=img_dir,
            img_ext=img_ext,
            subject_ids=subject_ids,
            study_ids=study_ids,
            dicom_ids=dicom_ids,
        )
    elif "chexpertplus" in img_dir:
        image_paths = get_chexpertplus_paths(
            split=split,
            img_dir=img_dir,
            img_ext=img_ext,
            subject_ids=subject_ids,
            study_ids=study_ids,
            dicom_ids=dicom_ids,
        )
    else:
        raise ValueError(
            f"Cannot infer from image directory path ({img_dir}) which "
            f"dataset style to use for getting image paths."
        )
    return image_paths
