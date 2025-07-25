import os
import sqlite3

import pandas as pd
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from wandb.util import generate_id

from remix.datasets import SECTION_T, CXRDataset
from remix.models import (
    ALIGNMENT_T,
    ImageTextMultiScaleContraster,
    ImageTextMultiScaleContrasterConfig,
)
from remix.utils import CXRTokenizer

torch.set_float32_matmul_precision("medium")


class LitData(LightningDataModule):
    def __init__(
        self,
        model_path: str,
        img_dir: str,
        img_ext: str,
        splits_path: str,
        notes_path: str,
        section: SECTION_T,
        metadata_path: str,
        frontal_only: bool,
        one_image_per_study: bool,
        num_chunks: int,
        num_overlap: int,
        mlm_probability: float,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.splits_df = pd.read_csv(self.hparams.splits_path)
        self.notes_df = pd.read_csv(self.hparams.notes_path)
        self.metadata_df = pd.read_csv(self.hparams.metadata_path)

        config = ImageTextMultiScaleContrasterConfig.from_pretrained(
            self.hparams.model_path,
        )
        self.tok = CXRTokenizer.from_pretrained(self.hparams.model_path)
        self.tok.model_max_length = config.max_position_embeddings

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = CXRDataset(
                img_dir=self.hparams.img_dir,
                img_ext=self.hparams.img_ext,
                splits_df=self.splits_df,
                split="train",
                notes_df=self.notes_df,
                section=self.hparams.section,
                metadata_df=self.metadata_df,
                frontal_only=self.hparams.frontal_only,
                one_image_per_study=self.hparams.one_image_per_study,
                num_chunks=self.hparams.num_chunks,
                num_overlap=self.hparams.num_overlap,
                text_tokenizer=self.tok,
                mlm_probability=self.hparams.mlm_probability,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )


class LitModel(LightningModule):
    def __init__(
        self,
        model_path: str,
        loss_combo: ALIGNMENT_T,
        checkpoint_path: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        config = ImageTextMultiScaleContrasterConfig.from_pretrained(
            self.hparams.model_path,
            loss_combo=self.hparams.loss_combo,
        )
        self.model = ImageTextMultiScaleContraster.from_pretrained(
            self.hparams.model_path,
            config=config,
        )

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch)
        self.log("train_loss", loss)
        return loss

    def on_train_end(self):
        self.model.save_pretrained(self.hparams.checkpoint_path)


@rank_zero_only
def makedirs_wrapper(save_dir, save_link):
    os.makedirs(save_dir)
    if os.path.exists(save_link):
        os.remove(save_link)
    rel_dir = save_dir
    rel_dir = rel_dir.replace(os.path.dirname(rel_dir), "")
    rel_dir = rel_dir.lstrip(os.path.sep)
    os.symlink(rel_dir, save_link)


class StrictWandbLogger(WandbLogger):
    def __init__(self, *, project: str, name: str, version: str, save_dir: str):
        self.best_link = os.path.join(save_dir, name, version + ".ckpt")
        save_link = os.path.join(save_dir, name, version)

        # make version unique
        version = version + "-" + generate_id()
        save_dir = os.path.join(save_dir, name, version)

        super().__init__(project=project, name=name, version=version, save_dir=save_dir)
        if os.path.exists(self.save_dir):
            raise FileExistsError(
                "\033[91mREAD THIS ERROR MSG: \033[0m"
                f"Experiment already exists at {self.save_dir}."
                " This logger uses some custom logic to put all logs,"
                " checkpoints, and configs related to an experiment"
                " under one directory. Please delete or rename to retry."
            )
        makedirs_wrapper(self.save_dir, save_link)

    def after_save_checkpoint(self, checkpoint_callback):
        best_model_path = checkpoint_callback.best_model_path
        best_model_path = best_model_path.replace(os.path.dirname(self.best_link), "")
        best_model_path = best_model_path.lstrip(os.path.sep)
        if os.path.exists(self.best_link):
            os.remove(self.best_link)
        os.symlink(
            best_model_path,
            self.best_link,
        )


class CustomCLI(LightningCLI):
    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler=None):
        if lr_scheduler is None:
            max_epochs = lightning_module.trainer.max_epochs
            num_training_steps = lightning_module.trainer.estimated_stepping_batches
            # single epoch warmup
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_training_steps // max_epochs,
                num_training_steps=num_training_steps,
            )
            lr_scheduler = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "lr",
            }
        return super(CustomCLI, CustomCLI).configure_optimizers(
            lightning_module,
            optimizer,
            lr_scheduler,
        )


def run():
    cli = CustomCLI()


if __name__ == "__main__":
    run()
