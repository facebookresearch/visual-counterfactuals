# Making Heads or Tails: Towards Semantically Consistent Visual Counterfactuals

This repository provides an implementation of the method from our ECCV 2022 [paper](https://arxiv.org/pdf/2203.12892) to compute semantically consistent visual counterfactuals.

## Setup

To setup a conda environment run

```shell
conda create -n counterfactuals python==3.8
conda activate counterfactuals
conda install pytorch torchvision torchaudio -c pytorch
conda install yaml
pip install pytorch-lightning
pip install -U albumentations
```

To use the code, you need to download and extract the [CUB-200-2011](https://data.caltech.edu/records/20098) dataset manually. Next, you need to set the dataset and output paths in the `utils/path.py` file.

## Training

For a quick start, download a model trained on CUB using

```shell
# Download files
wget -L https://dl.fbaipublicfiles.com/visual_counterfactuals/cub_res50_model.ckpt
wget -L https://dl.fbaipublicfiles.com/visual_counterfactuals/cub_vgg16_model.ckpt

# Place files under your output path specified in `utils/path.py`
# These paths are created automatically when training the model.
mkdir $OUTPUT_PATH
mkdir $OUTPUT_PATH/class_prediction_model_cub_res50
mkdir $OUTPUT_PATH/class_prediction_model_cub_vgg16
mv cub_res50_model.ckpt $OUTPUT_PATH/class_prediction_model_cub_res50/best_model.ckpt
mv cub_vgg16_model.ckpt $OUTPUT_PATH/class_prediction_model_cub_vgg16/best_model.ckpt
```

Alternatively, you can train an image classifier yourself

```shell
# Train a VGG-16 classifier on CUB
python class_prediction_model.py --config_path counterfactuals/configs/class_prediction_model/class_prediction_model_cub_vgg16.yaml

# Train a ResNet-50 classifier on CUB
python class_prediction_model.py --config_path counterfactuals/configs/class_prediction_model/class_prediction_model_cub_res50.yaml
```

## Counterfactuals

Run the following commands to generate counterfactual explanations via different methods. The [baseline](https://arxiv.org/pdf/1904.07451.pdf) uses the implementation from Goyal et al. Further, we compute explanations via our method.

```shell
# Generate explanations for a VGG-16 CUB classifier via Goyal et al.
python explain_model.py --config_path configs/counterfactuals/counterfactuals_goyal_cub_vgg16.yaml

# Generate explanations for a ResNet-50 CUB classifier via Goyal et al.
python explain_model.py --config_path configs/counterfactuals/counterfactuals_goyal_cub_res50.yaml

# Generate explanations for a VGG-16 classifier via our method
python explain_model.py --config_path configs/counterfactuals/counterfactuals_ours_cub_vgg16.yaml

# Generate explanations for ResNet-50 classifier via our method
python explain_model.py --config_path configs/counterfactuals/counterfactuals_ours_cub_res50.yaml
```

## Results

We obtain the following results with VGG-16 for __all edits__. Small differences with the results from the paper can be attributed to the variance across different CUB model training runs.

| Method     | Near KP | Same KP | Number Edits |
|------------|---------|---------|--------------|
| Baseline   | 57.6    | 8.8     | 5.4          |
| Ours       | 72.0    | 36.5    | 3.8          |

We obtain the following results with ResNet-50 for __all edits__.

| Method     | Near KP | Same KP | Number Edits |
|------------|---------|---------|--------------|
| Baseline   | 54.0    | 7.4     | 3.5          |
| Ours       | 64.6    | 31.1    | 2.9          |


## Visualization

You can visualize some counterfactual explanations by running the command below. You can update the `demo.py` code to visualize other examples. Each row shows a counterfactual edit. The first (top) row shows the first edit, and the last (bottom) row shows the last edit. After the last edit, the model's decision changed to the distractor class.

```shell
python demo.py --config_path configs/counterfactuals/counterfactuals_ours_cub_vgg16.yaml
```


## References

```bibtex
@inproceedings{vandenhende2022making,
  title={Making Heads or Tails: Towards Semantically Consistent Visual Counterfactuals},
  author={Vandenhende, Simon and Mahajan, Dhruv and Radenovic, Filip and Ghadiyaram, Deepti},
  booktitle={ECCV 2022},
  year={2022}
}
```

## License
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
