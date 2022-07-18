# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import yaml

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils.common_config import (
    get_model,
    get_test_dataset,
    get_test_transform,
    get_train_dataset,
    get_train_transform,
)
from utils.path import Path


parser = argparse.ArgumentParser(description="Train image classification model")
parser.add_argument("--config_path", type=str, required=True)


class ClassPredictionModule(pl.LightningModule):
    """
    Module for training image classifier.
    """

    def __init__(
        self,
        model,
        optimizer,
        learning_rate,
        momentum,
        weight_decay,
        milestones,
        gamma,
    ):
        super().__init__()
        self.save_hyperparameters(
            "optimizer",
            "learning_rate",
            "momentum",
            "weight_decay",
            "milestones",
            "gamma",
        )

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.acc1 = torchmetrics.Accuracy(top_k=1)
        self.acc5 = torchmetrics.Accuracy(top_k=5)

    def training_step(self, batch, batch_idx, key="train"):
        im_batch = batch["image"]
        classes_batch = batch["target"]

        outputs = self.model(im_batch)["logits"]
        loss = self.criterion(outputs, classes_batch)

        self.log(
            f"{key}_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )
        self.log(
            f"{key}_acc_1",
            self.acc1(outputs, classes_batch),
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )
        self.log(
            f"{key}_acc_5",
            self.acc5(outputs, classes_batch),
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, key="val"):
        im_batch = batch["image"]
        classes_batch = batch["target"]

        outputs = self.model(im_batch)["logits"]
        loss = self.criterion(outputs, classes_batch)

        self.log(
            f"{key}_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )
        self.log(
            f"{key}_acc_1",
            self.acc1(outputs, classes_batch),
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )
        self.log(
            f"{key}_acc_5",
            self.acc5(outputs, classes_batch),
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        parameters = self.model.parameters()

        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            parameters,
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.hparams.milestones,
            gamma=self.hparams.gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }


def main():
    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)

    experiment_name = os.path.basename(args.config_path).split(".")[0]
    dirpath = os.path.join(Path.output_root_dir(), experiment_name)

    tb_logger = TensorBoardLogger(dirpath, name="tb_logs")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc_1",
        dirpath=dirpath,
        filename="best_model",
        save_top_k=1,
        save_last=True,
        mode="max",
    )

    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        **config["trainer"],
    )

    train_dataset = get_train_dataset(transform=get_train_transform())
    val_dataset = get_test_dataset(transform=get_test_transform())

    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    model = ClassPredictionModule(
        get_model(config), **config["class_prediction_module"]
    )

    ckpt_last = os.path.join(checkpoint_callback.dirpath, "last.ckpt")
    ckpt_path = ckpt_last if os.path.isfile(ckpt_last) else None

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    result = trainer.validate(model, dataloaders=datamodule, ckpt_path="best")[0]

    with open(os.path.join(dirpath, "val.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
