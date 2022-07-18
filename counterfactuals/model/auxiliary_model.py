# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def get_auxiliary_model():
    # swav / deepcluster model yield same results on eval metrics
    # you can look for other ssl models online
    model = torch.hub.load("facebookresearch/swav", "resnet50")
    model.avgpool = torch.nn.Identity()
    model.fc = torch.nn.Identity()
    model.layer4 = model.layer4[0]  # only use conv5.1 in last block.
    dim = 2048
    n_pix = 7

    return model, dim, n_pix


@torch.inference_mode()
def process_dataset(model, dim, n_pix, dataloader, device):
    """
    Compute spatial feature representations using a pre-trained model.
    """
    model.to(device)
    model.eval()

    features = torch.FloatTensor(len(dataloader.dataset), dim, n_pix, n_pix)
    idx = 0

    for batch in dataloader:
        batch_size = batch.shape[0]
        output = model(batch.to(device)).reshape(batch_size, dim, n_pix, n_pix)
        features[idx : idx + batch_size].copy_(output)
        idx += batch_size

    return features
