# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import numpy as np
import torch


def compute_eval_metrics(
    counterfactuals,
    dataset,
):
    """
    Compute eval metrics from the paper `Counterfactual Visual Explanations`.

    We compute the Near-KP and Same-KP metric using the keypoint annotations
    from the dataset. In particular, for each cell replacement, we find what
    keypoints are present in the cell from the query and distractor image(s).
    If there are keypoints present, we increment the Near-KP score, which means
    we are replacing regions near important object parts. If the cell in the
    query image and the cell in the distractor image have at least one keypoint
    in common we increase the Same-KP metric. This means we are replacing
    semantically consistent parts. We track the metrics on a sample-wise level,
    and average the results when reporting numbers in the paper.

    Results are returned as a dictionary.
    """
    # initialize results dict
    results = {
        "Near-KP": defaultdict(list),
        "Same-KP": defaultdict(list),
    }

    # load keypoint annotations into memory once
    n_parts = len(dataset.parts_name_index)
    keypoints = torch.stack(
        [
            torch.from_numpy(dataset.__getitem__(index)["parts"])
            for index in range(len(dataset))
        ],
        dim=0,
    )  # n x n_parts x n_pix x n_pix (n_pix = 7)

    # evaluate all counterfactuals
    for counterfactual in counterfactuals.values():
        # gather query and distractor indexes
        query_index = counterfactual["query_index"]
        distractor_index = counterfactual["distractor_index"]

        # gather keypoint annotations for query and distractor images
        # of shape (n x n_parts x n_pix x n_pix)
        query_keypoints = keypoints[query_index]
        distractor_keypoints = keypoints[distractor_index]
        if len(query_keypoints.shape) == 3:
            query_keypoints = query_keypoints.unsqueeze(0)
        if len(distractor_keypoints.shape) == 3:
            distractor_keypoints = distractor_keypoints.unsqueeze(0)

        # flatten the keypoint maps
        query_keypoints = (
            torch.permute(query_keypoints, (1, 0, 2, 3)).reshape(n_parts, -1).float()
        )
        distractor_keypoints = (
            torch.permute(distractor_keypoints, (1, 0, 2, 3))
            .reshape(n_parts, -1)
            .float()
        )

        # measure near-kp and same-kp for each edit
        for edit in counterfactual["edits"]:
            query_cell, distractor_cell = edit[0], edit[1]
            near_kp = 0.5 * (
                torch.any(query_keypoints[:, query_cell]).float()
                + torch.any(distractor_keypoints[:, distractor_cell]).float()
            )
            same_kp = torch.any(
                torch.logical_and(
                    query_keypoints[:, query_cell],
                    distractor_keypoints[:, distractor_cell],
                )
            ).float()
            results["Near-KP"][query_index].append(near_kp.item())
            results["Same-KP"][query_index].append(same_kp.item())

    # average the results
    single_edit = {}
    all_edit = {}

    single_edit["Near-KP"] = np.nanmean([res[0] for res in results["Near-KP"].values()])
    single_edit["Same-KP"] = np.nanmean([res[0] for res in results["Same-KP"].values()])
    all_edit["Near-KP"] = np.nanmean(
        np.concatenate(list(res for res in results["Near-KP"].values()))
    )
    all_edit["Same-KP"] = np.nanmean(
        np.concatenate(list(res for res in results["Same-KP"].values()))
    )

    # report results for single-edit and all edits
    results["single_edit"] = single_edit
    results["all_edit"] = all_edit

    return results
