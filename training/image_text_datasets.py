import torch
from medcap.data.transforms import ResNet50Transform
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from transformers import PreTrainedTokenizer


class MIMIC_CXR_Dataset(Dataset):
    def __init__(
        self,
        *,
        image_paths: list[str],
        notes: list[str],
        chunks: list[list[str]],
        text_tokenizer: PreTrainedTokenizer,
    ):
        assert text_tokenizer.padding_side == "right"
        assert text_tokenizer.pad_token is not None
        assert text_tokenizer.model_max_length is not None

        self.image_paths = image_paths
        self.transform = ResNet50Transform()

        self.notes = notes
        self.chunks = chunks
        self.text_tokenizer = text_tokenizer

        assert len(set([len(image_paths), len(notes), len(chunks)])) == 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        note = self.notes[index]
        chunks = self.chunks[index]
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

        return {
            "input_ids_global": note_tok["input_ids"][0],
            "attention_mask_global": note_tok["attention_mask"][0],
            "input_ids_locals": chunks_tok["input_ids"],
            "attention_mask_locals": chunks_tok["attention_mask"],
            "images": im,
        }

    def __len__(self):
        return len(self.image_paths)
