# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import albumentations
import torch
import torchvision.transforms as transforms

from albumentations.pytorch.transforms import ToTensorV2
from data.cub import Cub
from model.model import ResNet50, VGG16


def get_model(p):
    if p["model"] == "VGG16":
        return VGG16(**p["model_kwargs"])

    elif p["model"] == "ResNet50":
        return ResNet50(**p["model_kwargs"])

    else:
        raise NotImplementedError


def get_train_dataset(transform, return_image_only=False):
    return Cub(train=True, transform=transform, return_image_only=return_image_only)


def get_test_dataset(transform, return_image_only=False):
    return Cub(train=False, transform=transform, return_image_only=return_image_only)


def get_train_transform():
    return albumentations.Compose(
        [
            albumentations.RandomResizedCrop(width=224, height=224),
            albumentations.HorizontalFlip(),
            albumentations.Normalize((0.471, 0.460, 0.454), (0.267, 0.266, 0.271)),
            ToTensorV2(),
        ],
        keypoint_params=albumentations.KeypointParams(
            format="xy", remove_invisible=True, label_fields=["keypoints_ids"]
        ),
    )


def get_vis_transform():
    return albumentations.Compose(
        [
            albumentations.SmallestMaxSize(max_size=256),
            albumentations.CenterCrop(width=224, height=224),
        ]
    )


def get_test_transform():
    return albumentations.Compose(
        [
            albumentations.SmallestMaxSize(max_size=256),
            albumentations.CenterCrop(width=224, height=224),
            albumentations.Normalize((0.471, 0.460, 0.454), (0.267, 0.266, 0.271)),
            ToTensorV2(),
        ],
        keypoint_params=albumentations.KeypointParams(
            format="xy", remove_invisible=True, label_fields=["keypoints_ids"]
        ),
    )


def get_imagenet_test_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_test_dataloader(p, dataset):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=p["num_workers"],
        batch_size=p["batch_size"],
        shuffle=False,
        drop_last=False,
    )
