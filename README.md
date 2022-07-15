# Making Heads or Tails: Towards Semantically Consistent Visual Counterfactuals

This repository provides an implementation of the method from our [paper](https://arxiv.org/pdf/2203.12892) to compute semantically consistent visual counterfactuals.

## Setup

To setup a conda environment run

```shell
conda create --name counterfactuals python=3.9
conda activate counterfactuals
pip install blobfile==1.2.7
pip install -r requirements.txt
pip install --no-deps -r requirements_no_deps.txt
```

To use the code, you need to download and extract the CUB-200-2011 dataset from [here](https://data.caltech.edu/records/20098) manually. Next, you need to set the `dataset_folder` in the `./config/class_prediction_base.yaml` and `./config/counterfactuals_base.yaml` files.

## Training

For a quick start, download a model trained on CUB using
```shell
wget -L https://dl.fbaipublicfiles.com/visual_counterfactuals/cub_res50_model.ckpt
```

Alternatively, to train a model on CUB run

```shell
python class_prediction_model.py
```

## Counterfactuals

First make sure the dataset folder and model path in the `./config/counterfactuals_base.yaml` file are set correctly. Next, you can run the following commands to compute counterfactuals via the [baseline](https://arxiv.org/abs/1904.07451) method or our method.

```shell
# run baseline method without semantic constraint
python explain_model.py

# run our method with semantic constraint (you can adjust the hyperparameters via the config file)
python explain_model.py explainer=soft_constraint
```

You should get the following results when running the code based on our pre-trained model. Note that the numbers for all methods in the table are higher than the ones reported in the paper. This is because we use the Albumentations library in the code to transform the keypoint locations with the image. This yields more precise localization compared to the implementation we got from Goyal et al., and which we used for the paper. Overall, the conclusions stay the same. You can play around with the hyperparameters of our algorithm by adapting the `./config/explainer/soft_constraint.yaml` file. The code will generate visualizations of the counterfactuals in the output directory. You can disable this option through the arguments of `explain_dataset()` in the main function.

| Method     | Near KP | Same KP | Number Edits |
|------------|---------|---------|--------------|
| Goyal      | 53.7    | 30.4    | 3.70         |
| Ours       | 62.4    | 46.8    | 3.50         |


## References

```bibtex
@article{vandenhende2022making,
  title={Making Heads or Tails: Towards Semantically Consistent Visual Counterfactuals},
  author={Vandenhende, Simon and Mahajan, Dhruv and Radenovic, Filip and Ghadiyaram, Deepti},
  journal={arXiv preprint arXiv:2203.12892},
  year={2022}
}
```

## License
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
