# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pathlib

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from torchvision.datasets.folder import default_loader
from utils.path import Path


class Cub(Dataset):
    """
    Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version
    of the CUB-200 dataset, with roughly double the number of images per
    class and new part location annotations. For detailed information
    about the dataset, please see the technical report linked below.
    Number of categories: 200
    Number of images: 11,788
    Annotations per image: 15 Part Locations
    Webpage: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    README: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/README.txt  # noqa
    Download: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
    """

    def __init__(
        self,
        train=True,
        transform=None,
        loader=default_loader,
        return_image_only=False,
    ):

        self._dataset_folder = pathlib.Path(Path.db_root_dir("CUB"))
        self._transform = transform
        self._loader = loader
        self._train = train
        self._class_name_index = {}
        self._return_image_only = return_image_only

        if not self._check_dataset_folder():
            raise RuntimeError(
                "Dataset not downloaded, download it from "
                "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"  # noqa
            )

    def _load_metadata(self):
        images = pd.read_csv(
            self._dataset_folder.joinpath("images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )

        image_class_labels = pd.read_csv(
            self._dataset_folder.joinpath("image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )

        train_eval_split = pd.read_csv(
            self._dataset_folder.joinpath("train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        with open(self._dataset_folder.joinpath("classes.txt")) as f:
            for line in f:
                class_label, class_name = line.strip().split(" ", 1)
                class_label = int(class_label) - 1
                self._class_name_index[class_label] = class_name

        # load parts information
        self._original_parts_name_index = {}

        with open(self._dataset_folder.joinpath("parts", "parts.txt")) as f:
            for line in f:
                cols = line.strip().split(" ", 1)
                assert len(cols) == 2
                part_id = int(cols[0]) - 1
                part_name = cols[1]
                self._original_parts_name_index[part_id] = part_name

        self._inverse_original_parts_name_index = {
            value: key for key, value in self._original_parts_name_index.items()
        }

        image_parts = pd.read_csv(
            self._dataset_folder.joinpath("parts", "part_locs.txt"),
            sep=" ",
            names=["img_id", "part_id", "x", "y", "visible"],
        )
        image_parts = image_parts[image_parts["visible"] == 1]
        image_parts = image_parts.groupby("img_id")[["part_id", "x", "y"]].agg(
            lambda x: list(x)
        )

        # define remapping for part instances to merge left-right instances
        parts_name_remap = {
            "back": "back",
            "beak": "beak",
            "belly": "belly",
            "breast": "breast",
            "crown": "crown",
            "forehead": "forehead",
            "left eye": "eye",
            "left leg": "leg",
            "left wing": "wing",
            "nape": "nape",
            "right eye": "eye",
            "right leg": "leg",
            "right wing": "wing",
            "tail": "tail",
            "throat": "throat",
        }

        self._parts_name_index = {
            0: "back",
            1: "beak",
            2: "belly",
            3: "breast",
            4: "crown",
            5: "forehead",
            6: "eye",
            7: "leg",
            8: "wing",
            9: "nape",
            10: "tail",
            11: "throat",
        }

        self._inverse_parts_name_index = {
            value: key for key, value in self._parts_name_index.items()
        }

        self._parts_index_remap = {
            self._inverse_original_parts_name_index[
                key
            ]: self._inverse_parts_name_index[value]
            for key, value in parts_name_remap.items()
        }

        # merge
        data = images.merge(image_class_labels, on="img_id")
        data = data.merge(image_parts, on="img_id")
        self._data = data.merge(train_eval_split, on="img_id")

        # select split
        if self._train:
            self._data = self._data[self._data.is_training_img == 1]
        else:
            self._data = self._data[self._data.is_training_img == 0]

    def _check_dataset_folder(self):
        try:
            self._load_metadata()
        except Exception as e:
            print(f"Error: {e}")
            return False

        for _, row in self._data.iterrows():
            filepath = self._dataset_folder.joinpath("images", row.filepath)
            if not pathlib.Path.exists(filepath):
                return False
        return True

    @property
    def class_name_index(self):
        return self._class_name_index

    @property
    def parts_name_index(self):
        return self._parts_name_index

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sample = self._data.iloc[idx]
        path = self._dataset_folder.joinpath("images", sample.filepath)

        image = self._loader(path)
        width, height = image.size

        # return image only
        if self._return_image_only:
            if self._transform is None:
                return image

            else:
                if "albumentations" in str(type(self._transform)):
                    return self._transform(image=np.array(image, dtype=np.uint8))[
                        "image"
                    ]
                else:
                    return self._transform(image)

        target = sample.target - 1

        # load parts
        part_ids = np.array(sample.part_id, dtype=np.int32) - 1
        part_locs = np.stack(
            (
                np.array(sample.x, dtype=np.float32),
                np.array(sample.y, dtype=np.float32),
            ),
            axis=1,
        )

        valid_x_coords = np.logical_and(part_locs[:, 0] > 0, part_locs[:, 0] < width)
        valid_y_coords = np.logical_and(part_locs[:, 1] > 0, part_locs[:, 1] < height)
        valid_coords = np.logical_and(valid_x_coords, valid_y_coords)
        part_ids = part_ids[valid_coords]
        part_locs = part_locs[valid_coords]

        # transform
        if self._transform is not None:
            sample = self._transform(
                image=np.array(image, dtype=np.uint8),
                keypoints=part_locs,
                keypoints_ids=part_ids,
            )
            sample = {
                "image": sample["image"],
                "part_locs": np.array(sample["keypoints"]),
                "part_ids": np.array(sample["keypoints_ids"]),
            }

        else:
            sample = {
                "image": image,
                "part_locs": part_locs,
                "part_ids": part_ids,
            }

        # return parts as binary mask on 7 x 7 grid to ease evaluation
        part_locs = sample["part_locs"]
        part_ids = sample["part_ids"]
        n_pix_per_cell_h = sample["image"].shape[1] // 7
        n_pix_per_cell_w = sample["image"].shape[2] // 7
        parts = np.zeros((len(self.parts_name_index), 7, 7), dtype=np.uint8)
        for part_loc, part_id in zip(part_locs, part_ids):
            x_coord = int(part_loc[0] // n_pix_per_cell_w)
            y_coord = int(part_loc[1] // n_pix_per_cell_h)
            new_part_id = self._parts_index_remap[part_id]
            parts[new_part_id, y_coord, x_coord] = 1

        output = {
            "image": sample["image"],
            "target": target,
            "parts": parts,
        }

        return output

    def get_target(self, target):
        return (
            np.argwhere(np.array(self._data["target"].tolist()) == target + 1)
            .reshape(-1)
            .tolist()
        )
