import os

import lightning as L
import pandas as pd
import torch
from image_text_datasets import MIMIC_CXR_Dataset
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from utils import mimic_cxr_get_single_image_per_study

from models import (
    ImageTextMultiScaleContraster,
    ImageTextMultiScaleContrasterConfig,
)

torch.set_float32_matmul_precision("medium")

num_epochs = 10
num_workers = 4

batch_size = 22
grad_accum = 1

lr = 1e-5

base_model_path = "microsoft/BiomedVLP-CXR-BERT-specialized"
data_path = "/opt/gpudata/imadejski/mimic-cxr"

output_path = "/opt/gpudata/imadejski/image_text_global_local_alignment"
checkpoint_path = os.path.join(output_path, "model")

config = ImageTextMultiScaleContrasterConfig.from_pretrained(base_model_path)

tok = BertTokenizer.from_pretrained(base_model_path)
tok.model_max_length = config.max_position_embeddings

df = pd.read_csv(
    os.path.join(data_path, "mimic_cxr_train_sentence_chunk_sectioned.csv")
)
image_paths = mimic_cxr_get_single_image_per_study(df["study"].to_list(), data_path)
ds = MIMIC_CXR_Dataset(
    image_paths=image_paths,
    notes=df["combined"].to_list(),
    chunks=df.iloc[:, 2:].values.tolist(),
    text_tokenizer=tok,
)
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

model = ImageTextMultiScaleContraster.from_pretrained(
    base_model_path,
    config=config,
)
model.resnet.load_biovil_weights()

# Check if the model is in training mode
if not model.training:
    model.train()
    print("Model is now in training mode.")


class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=lr)
        num_training_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps,
        )
        lr_scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "name": "lr",
        }
        return [optimizer], [lr_scheduler]


lr_monitor = LearningRateMonitor(logging_interval="step")
callbacks = [
    lr_monitor,
]

lit_model = LitModel(model)

trainer = L.Trainer(
    accelerator="gpu",
    accumulate_grad_batches=grad_accum,
    max_epochs=num_epochs,
    precision="16-mixed",
    logger=True,
    num_sanity_val_steps=0,
    default_root_dir=output_path,
    callbacks=callbacks,
    log_every_n_steps=1,
    # limit_train_batches=10,
)
trainer.fit(model=lit_model, train_dataloaders=dl)
model.save_pretrained(checkpoint_path)
