# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import numpy as np
import torch
import yaml

import model.auxiliary_model as auxiliary_model

from explainer.counterfactuals import compute_counterfactual
from explainer.eval import compute_eval_metrics
from explainer.utils import get_query_distractor_pairs, process_dataset
from utils.common_config import (
    get_imagenet_test_transform,
    get_model,
    get_test_dataloader,
    get_test_dataset,
    get_test_transform,
)
from utils.path import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Generate counterfactual explanations"
)
parser.add_argument(
    "--config_path", type=str, required=True
)


def main():
    args = parser.parse_args()

    # parse args
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)

    experiment_name = os.path.basename(args.config_path).split(".")[0]
    dirpath = os.path.join(Path.output_root_dir(), experiment_name)
    os.makedirs(dirpath, exist_ok=True)

    # create dataset
    dataset = get_test_dataset(
        transform=get_test_transform()
    )
    dataloader = get_test_dataloader(config, dataset)

    # device
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    # load classifier
    print("Load classification model weights")
    model = get_model(config)
    model_path = os.path.join(
        Path.output_root_dir(),
        config["counterfactuals_kwargs"]["model"],
    )
    state_dict = torch.load(model_path)["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key[len("model."):]] = state_dict[key]
        del state_dict[key]
    model.load_state_dict(state_dict, strict=True)

    # process dataset
    print("Pre-compute classifier predictions")
    result = process_dataset(model, dataloader, device)
    features = result["features"]
    preds = result["preds"].numpy()
    targets = result["targets"].numpy()
    print("Top-1 accuracy: {:.2f}".format(100 * result["top1"]))

    # compute query-distractor pairs
    print("Pre-compute query-distractor pairs")
    query_distractor_pairs = get_query_distractor_pairs(
        dataset,
        confusion_matrix=result["confusion_matrix"],
        max_num_distractors=config["counterfactuals_kwargs"]["max_num_distractors"]  # noqa
    )

    # get classifier head
    classifier_head = model.get_classifier_head()
    classifier_head = torch.nn.DataParallel(classifier_head.cuda())
    classifier_head.eval()

    # auxiliary features for soft constraint
    if config["counterfactuals_kwargs"]["apply_soft_constraint"]:
        print("Pre-compute auxiliary features for soft constraint")
        aux_model, aux_dim, n_pix = auxiliary_model.get_auxiliary_model()
        aux_transform = get_imagenet_test_transform()
        aux_dataset = get_test_dataset(
            transform=aux_transform, return_image_only=True
        )
        aux_loader = get_test_dataloader(config, aux_dataset)

        auxiliary_features = auxiliary_model.process_dataset(
            aux_model,
            aux_dim,
            n_pix,
            aux_loader,
            device,
        ).numpy()
        use_auxiliary_features = True

    else:
        use_auxiliary_features = False

    # compute counterfactuals
    print("Compute counterfactuals")
    counterfactuals = {}

    for query_index in tqdm(range(len(dataset))):
        if query_index not in query_distractor_pairs.keys():
            continue  # skips images that were classified incorrectly

        # gather query features
        query = features[query_index]  # dim x n_row x n_row
        query_pred = preds[query_index]
        if query_pred != targets[query_index]:
            continue  # skip if query classified incorrect

        # gather distractor features
        distractor_target = query_distractor_pairs[query_index]["distractor_class"]  # noqa
        distractor_index = query_distractor_pairs[query_index]["distractor_index"]  # noqa
        if isinstance(distractor_index, int):
            if preds[distractor_index] != distractor_target:
                continue  # skip if distractor classified is incorrect
            distractor_index = [distractor_index]

        else:  # list
            distractor_index = [
                jj for jj in distractor_index if preds[jj] == distractor_target
            ]
            if len(distractor_index) == 0:
                continue  # skip if no distractors classified correct

        distractor = torch.stack(
            [features[jj] for jj in distractor_index], dim=0
        )

        # soft constraint uses auxiliary features
        if use_auxiliary_features:
            query_aux_features = torch.from_numpy(
                auxiliary_features[query_index]
            )  # aux_dim x n_row x n_row
            distractor_aux_features = torch.stack(
                [
                    torch.from_numpy(auxiliary_features[jj])
                    for jj in distractor_index
                ],
                dim=0,
            )  # n x aux_dim x n_row x n_row

        else:
            query_aux_features = None
            distractor_aux_features = None

        # compute counterfactual
        try:
            list_of_edits = compute_counterfactual(
                query=query,
                distractor=distractor,
                classification_head=classifier_head,
                distractor_class=distractor_target,
                query_aux_features=query_aux_features,
                distractor_aux_features=distractor_aux_features,
                lambd=config["counterfactuals_kwargs"]["lambd"],
                temperature=config["counterfactuals_kwargs"]["temperature"],
                topk=config["counterfactuals_kwargs"]["topk"]
                if "topk" in config["counterfactuals_kwargs"].keys()
                else None,
            )

        except BaseException:
            print("warning - no counterfactual @ index {}".format(query_index))
            continue

        counterfactuals[query_index] = {
            "query_index": query_index,
            "distractor_index": distractor_index,
            "query_target": query_pred,
            "distractor_target": distractor_target,
            "edits": list_of_edits,
        }

    # save result
    np.save(os.path.join(dirpath, "counterfactuals.npy"), counterfactuals)

    # evaluation
    print("Generated {} counterfactual explanations".format(
        len(counterfactuals)
    ))
    average_num_edits = np.mean(
        [len(res["edits"]) for res in counterfactuals.values()]
    )
    print("Average number of edits is {:.2f}".format(average_num_edits))

    result = compute_eval_metrics(
        counterfactuals,
        dataset=dataset,
    )

    print("Eval results single edit: {}".format(result["single_edit"]))
    print("Eval results all edits: {}".format(result["all_edit"]))


if __name__ == "__main__":
    main()
