# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_counterfactual(
    query,
    distractor,
    classification_head,
    distractor_class,
    query_aux_features,
    distractor_aux_features,
    lambd,
    temperature,
    topk,
):
    """
    args:
        query: query feature repr (dim x n_pix x n_pix)
        distractor: distractor feature repr (N x dim x n_pix x n_pix)
        classification_head: classification head
        distractor_class: distractor class
        query_aux_features: query auxiliary feature repr (dim_aux x n_pix x n_pix)
        distractor_aux_features: distractor auxiliary feature repr
            (N x dim_aux x n_pix x n_pix)
        lambd: lambda loss balancing weight
        topk: only use top-k most similar cells

    return:
        edits: list of edits that flip model's prediction
    """
    # eval
    classification_head.eval()

    # get dimensions of spatial feature representations
    n_feat, n_pix = query.shape[0], query.shape[1]
    n_pixels = n_pix * n_pix

    # flatten
    query_fl = query.reshape(n_feat, -1).t().cuda()  # n_pixels x dim
    distractor_fl = (
        torch.permute(distractor, (0, 2, 3, 1)).reshape(-1, n_feat).cuda()
    )  # N * n_pixels x dim

    # flatten aux features
    if query_aux_features is not None and distractor_aux_features is not None:
        query_aux_features_fl = query_aux_features.reshape(
            query_aux_features.shape[0], -1
        ).t()  # n_pixels x aux_dim
        distractor_aux_features_fl = torch.permute(
            distractor_aux_features, (0, 2, 3, 1)
        ).reshape(
            -1, query_aux_features_fl.shape[1]
        )  # N * n_pixels x aux_dim

    else:
        query_aux_features_fl, distractor_aux_features_fl = None, None

    # determine all possible edits
    query_edits = (
        np.arange(query_fl.shape[0] * distractor_fl.shape[0]) // distractor_fl.shape[0]
    )
    distractor_edits = (
        np.arange(query_fl.shape[0] * distractor_fl.shape[0]) % distractor_fl.shape[0]
    )
    all_edits = [(i, j) for i, j in zip(query_edits, distractor_edits)]

    if topk is not None:
        all_edits = _find_knn_cells(
            query_aux_features_fl, distractor_aux_features_fl, all_edits, topk
        )

    # init variables
    current = query_fl.clone()
    list_of_edits = []
    all_combinations = _get_feature_representations_of_all_edits(
        query_fl, distractor_fl, all_edits
    )

    # loop until prediction changes
    while (
        torch.argmax(
            classification_head(current.t().contiguous().view(1, n_feat, n_pix, n_pix)),
            dim=1,
        )
        != distractor_class
    ):
        # find next best cell replacement
        query_cell, distractor_cell = _find_single_best_edit(
            all_combinations,
            all_edits,
            distractor_class,
            classification_head,
            query_aux_features_fl,
            distractor_aux_features_fl,
            lambd=lambd,
            dims=(n_feat, n_pix, n_pixels),
            temperature=temperature,
        )

        # update variables
        list_of_edits.append((query_cell, distractor_cell))
        current[query_cell].copy_(distractor_fl[distractor_cell])
        all_edits = [
            (i, j)
            for i, j in all_edits
            if i != query_cell and j != distractor_cell  # noqa
        ]
        all_combinations = _get_feature_representations_of_all_edits(
            current, distractor_fl, all_edits
        )

    return list_of_edits


def _find_knn_cells(query_aux_features, distractor_aux_features, all_edits, topk):
    """
    Find top-K most semantically similar cells using auxiliary feature
    representations.
    """
    assert len(query_aux_features.shape) == 2
    assert len(distractor_aux_features.shape) == 2

    query_aux_features = F.normalize(
        query_aux_features, dim=1, p=2
    )  # n_pixels x dim_aux
    distractor_aux_features = F.normalize(
        distractor_aux_features, dim=1, p=2
    )  # n_pixels x dim_aux

    logits = torch.matmul(query_aux_features, distractor_aux_features.t())
    logits = torch.FloatTensor([logits[edit] for edit in all_edits])
    numel = int(torch.numel(logits) * topk)
    knn = torch.topk(logits, k=numel, largest=True)[1].tolist()

    return [all_edits[ii] for ii in knn]


def _get_feature_representations_of_all_edits(query_fl, distractor_fl, all_edits):
    """
    Construct feature representations when performing different edits.
    """
    num_allowed = len(all_edits)
    features_all_edits = torch.clone(query_fl).repeat(num_allowed, 1, 1)

    for ii in range(num_allowed):
        cell_I, cell_I_prime = all_edits[ii]
        features_all_edits[ii, cell_I].copy_(distractor_fl[cell_I_prime])

    return features_all_edits


@torch.no_grad()
def _find_single_best_edit(
    all_combinations,
    all_edits,
    distractor_class,
    classification_head,
    query_aux_features,
    distractor_aux_features,
    lambd,
    dims,
    temperature,
):
    """
    Find next single best edit.

    args:
        all_combinations: all combinations of query and distractor features
            when applying different single edits.
        all_edits: all edits of query and distractor cells
        distractor_class: the distractor class index
        classification_head: the classification head
        query_aux_features: auxiliary features query
        distractor_aux_features: auxiliary features distractor
        lambd: weight to balance the losses
        dims: features_dim, n_pix, n_pixels
        temperature: softmax temperature

    return:
        edit_query, edit_distractor: single best edit based on loss objective
        from the paper (Equation 4).

        argmax class_objective(edit) + lambd * semantic_objective(edit)
    """
    # compute classification loss via classifier head
    classification_head.eval()
    n_feat, n_pix, _ = dims

    logits_class = classification_head(
        torch.transpose(all_combinations, 1, 2)
        .contiguous()
        .view(-1, n_feat, n_pix, n_pix)
        .cuda()
    )
    probs_class = F.softmax(logits_class, dim=1)[:, distractor_class]
    optim_class = probs_class.cpu().numpy()
    optim_class = np.log(optim_class)

    # determine semantic similarity loss via auxiliary model
    if lambd > 0:
        # normalize auxiliary feature representations
        query_aux_features = F.normalize(
            query_aux_features, dim=1, p=2
        )  # n_pixels x dim_aux
        distractor_aux_features = F.normalize(
            distractor_aux_features, dim=1, p=2
        )  # n_pixels x dim_aux

        # compute dot product (cosine similarity)
        logits = torch.matmul(query_aux_features, distractor_aux_features.t())

        # use temperature
        logits /= temperature

        # non-parametric softmax
        probs = F.softmax(logits, dim=1)

        # numpy
        probs = probs.cpu().numpy()

        # select edits
        optim_consistency = np.array([probs[v] for v in all_edits])

        # log-space
        optim_consistency = np.log(optim_consistency)

        optim_total = optim_class + lambd * optim_consistency

    else:
        optim_total = optim_class

    # find best edit
    best_edit = np.argmax(optim_total)

    return all_edits[best_edit][0], all_edits[best_edit][1]
