"""Computer vision example on Transfer Learning. This computer vision example illustrates how one could fine-tune a
pre-trained network (by default, a ResNet50 is used) using pytorch-lightning. For the sake of this example, the
'cats and dogs dataset' (~60MB, see `DATA_URL` below) and the proposed network (denoted by `TransferLearningModel`,
see below) is trained for 15 epochs.
The training consists of three stages.
From epoch 0 to 4, the feature extractor (the pre-trained network) is frozen except
maybe for the BatchNorm layers (depending on whether `train_bn = True`). The BatchNorm
layers (if `train_bn = True`) and the parameters of the classifier are trained as a
single parameters group with lr = 1e-2.
From epoch 5 to 9, the last two layer groups of the pre-trained network are unfrozen
and added to the optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3
for the first parameter group in the optimizer).
Eventually, from epoch 10, all the remaining layer groups of the pre-trained network
are unfrozen and added to the optimizer as a third parameter group. From epoch 10,
the parameters of the pre-trained network are trained with lr = 1e-5 while those of
the classifier is trained with lr = 1e-4.
Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
To run:
    python computer_vision_fine_tuning.py fit
"""

from audioop import cross
import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from torchvision import models, transforms

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities.cli import LightningCLI

from dataloader import NeutDataset
from functools import partial

log = logging.getLogger(__name__)

#  --- Finetuning Callback ---
def create_three_channels(image):
    return np.stack(arrays=(image, image, image), axis=2)


def normalize_transform(image):
    return image / 255.0


class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: tuple = (1, 2), train_bn: bool = False):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ):
        if epoch == self.milestones[0]:
            # unfreeze 5 last layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )

        elif epoch == self.milestones[1]:
            # unfreeze remaining layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )


class NeutDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 8,
    ):
        """CatDogImageDataModule.
        Args:
            num_workers: number of CPU workers
            batch_size: number of sample in a batch
        """
        super().__init__()

        self._num_workers = num_workers
        self._batch_size = batch_size

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Lambda(create_three_channels),
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose(
            [
                transforms.Lambda(create_three_channels),
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def create_dataset(self, split, transform=None):
        return NeutDataset(transform=transform, split=split)

    def __dataloader(self, train: bool):
        """Train/validation loaders."""
        if train:
            dataset = self.create_dataset(transform=self.train_transform, split="train")
        else:
            dataset = self.create_dataset(
                transform=self.valid_transform, split="validation"
            )
        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=train,
        )

    def train_dataloader(self):
        log.info("Training data loaded.")
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info("Validation data loaded.")
        return self.__dataloader(train=False)


#  --- Pytorch-lightning module ---


class TransferLearningModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "vgg16",
        train_bn: bool = False,
        milestones: tuple = (1, 2),
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            backbone: Name (as in ``torchvision.models``) of the feature extractor
            train_bn: Whether the BatchNorm layers should be trainable
            milestones: List of two epochs milestones
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
        """
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        self.__build_model()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.val_f1_score = F1Score()
        self.save_hyperparameters()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)

        _layers = list(backbone.children())
        self.feature_extractor = nn.Sequential(*_layers)[:-1]

        self.anti = nn.Sequential(nn.Linear(in_features=32, out_features=2))
        self.flavor = nn.Sequential(nn.Linear(in_features=32, out_features=4))
        self.interaction = nn.Sequential(nn.Linear(in_features=32, out_features=4))
        self.protons = nn.Sequential(nn.Linear(in_features=32, out_features=4))
        self.pions = nn.Sequential(nn.Linear(in_features=32, out_features=4))
        self.pizeros = nn.Sequential(nn.Linear(in_features=32, out_features=4))
        self.neutrons = nn.Sequential(nn.Linear(in_features=32, out_features=4))
        # 2. Classifier:
        _fc_layers = [
            nn.Linear(25088, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        ]
        self.fc = nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.cross_entropy = partial(F.cross_entropy, ignore_index=-1)

    def forward(self, x):
        """Forward pass.
        Returns logits.
        """

        # 1. Feature extraction:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        outputs = {
            "anti": self.anti(x),
            "flavor": self.flavor(x),
            "interaction": self.interaction(x),
            "protons": self.protons(x),
            "pions": self.pions(x),
            "pizeros": self.pizeros(x),
            "neutrons": self.neutrons(x),
        }
        return outputs

    def loss(self, outputs, y):
        loss = torch.tensor(0.0, device="cuda:0")
        output_names = list(outputs)
        for idx, output in enumerate(output_names):
            loss += self.cross_entropy(outputs[output], y[:, idx])
        return loss

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        outputs = self.forward(x)

        # 2. Compute loss
        train_loss = self.loss(outputs, y)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(outputs["flavor"], y[:, 1]), prog_bar=True)
        return train_loss

    def on_epoch_end(self):
        self.log('train_acc_epoch', self.train_acc.compute())
        self.log('val_acc_epoch', self.valid_acc.compute())
        self.train_acc.reset()
        self.valid_acc.reset()

        self.log('val_f1_epoch', self.val_f1_score.compute())
        self.val_f1_score.reset()

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        outputs = self.forward(x)

        # 2. Compute loss
        self.log("val_acc", self.valid_acc(outputs["flavor"], y[:, 1]), prog_bar=True)
        self.log(
            "val_f1score", self.val_f1_score(outputs["flavor"], y[:, 1]), prog_bar=True
        )
        val_loss = self.loss(outputs, y)
        return val_loss

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        print(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma
        )
        return [optimizer], [scheduler]


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MilestonesFinetuning, "finetuning")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("finetuning.milestones", "model.milestones")
        parser.link_arguments("finetuning.train_bn", "model.train_bn")
        parser.set_defaults(
            {
                "trainer.max_epochs": 15,
                "trainer.enable_model_summary": True,
                "trainer.num_sanity_val_steps": 0,
                "trainer.gpus": 1,
                "trainer.default_root_dir": "/afs/cern.ch/work/r/rradev/public/vgg_cvn/logs",
            }
        )


def cli_main():
    MyLightningCLI(TransferLearningModel, NeutDataModule, seed_everything_default=1234)


if __name__ == "__main__":
    cli_main()
