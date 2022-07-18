# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchvision.models


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        vgg = torchvision.models.__dict__["vgg16_bn"](pretrained=True)
        vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, num_classes)

        self.vgg = vgg
        self.num_classes = num_classes

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return {"features": features, "logits": logits}

    def features(self, x):
        return self.vgg.features(x)

    def classifier(self, x):
        return self.vgg.classifier(x.view(x.size(0), -1))

    def get_classifier_head(self):
        return nn.Sequential(nn.Flatten(start_dim=1), self.vgg.classifier)

    def get_features_dim(self):
        return {"n_feat": 512, "n_row": 7, "n_pixels": 49}

    def get_num_classes(self):
        return self.num_classes


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        resnet = torchvision.models.__dict__["resnet50"](pretrained=True)
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        self.resnet = resnet
        self.num_classes = num_classes

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return {"features": features, "logits": logits}

    def features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4[0](x)
        return x

    def classifier(self, x):
        x = self.resnet.layer4[1:](x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

    def get_classifier_head(self):
        return nn.Sequential(
            self.resnet.layer4[1:],
            self.resnet.avgpool,
            nn.Flatten(start_dim=1),
            self.resnet.fc,
        )

    def get_features_dim(self):
        return {"n_feat": 2048, "n_row": 7, "n_pixels": 49}

    def get_num_classes(self):
        return self.num_classes
