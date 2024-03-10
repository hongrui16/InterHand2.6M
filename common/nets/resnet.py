
import os, sys
import os.path as osp
import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models.resnet import BasicBlock, Bottleneck
# from torchvision.models.resnet import model_urls

sys.path.append('../..')
from config import config as cfg

# class ResNetBackbone(nn.Module):
#     def __init__(self, resnet_type, pretrained = True):
#         super(ResNetBackbone, self).__init__()
#         # Dynamically load the pretrained ResNet model
#         self.model = getattr(models, f'resnet{resnet_type}')(pretrained=pretrained)
#         # Replace the fully connected layer with nn.Identity
#         self.model.fc = nn.Identity()

#     def forward(self, x):
#         # Forward pass through the modified ResNet model
#         return self.model(x)


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):
	
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) # RGB
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def init_weights(self):
        # Dynamically load the pretrained model
        org_resnet = getattr(models, self.name)(pretrained=True)
        
        # Get the state dictionary from the original pretrained model
        org_state_dict = org_resnet.state_dict()
        
        # Remove the fully connected layer parameters
        org_state_dict.pop('fc.weight', None)
        org_state_dict.pop('fc.bias', None)
        
        # Load the modified state dict into your custom model
        self.load_state_dict(org_state_dict, strict=False)




if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    img = torch.rand(1,3,256,256).to(device)
    backbone_net = ResNetBackbone(18).to(device)
    backbone_net.init_weights()
    img_feat = backbone_net(img)
    print(img_feat.shape) ### torch.Size([1, 2048])