# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import os, sys

sys.path.append('../')

from config import config as cfg
from utils.compute_heatmap import render_gaussian_heatmap

class JointHeatmapLoss(nn.Module):
    def __ini__(self):
        super(JointHeatmapLoss, self).__init__()

    def forward(self, joint_out, joint_gt, joint_valid):
        loss = (joint_out - joint_gt)**2 * joint_valid[:,:,None,None,None]
        return loss

class HandTypeLoss(nn.Module):
    def __init__(self):
        super(HandTypeLoss, self).__init__()

    def forward(self, hand_type_out, hand_type_gt, hand_type_valid):
        loss = F.binary_cross_entropy(hand_type_out, hand_type_gt, reduction='none')
        loss = loss.mean(1)
        loss = loss * hand_type_valid

        return loss

class RelRootDepthLoss(nn.Module):
    def __init__(self):
        super(RelRootDepthLoss, self).__init__()

    def forward(self, root_depth_out, root_depth_gt, root_valid):
        loss = torch.abs(root_depth_out - root_depth_gt) * root_valid
        return loss


class LossCalculation(nn.Module):
    def __init__(self, device = 'cpu'):
        super(LossCalculation, self).__init__()

        # loss functions
        self.joint_heatmap_loss = JointHeatmapLoss()
        self.rel_root_depth_loss = RelRootDepthLoss()
        self.hand_type_loss = HandTypeLoss()

    
    def forward(self, joint_heatmap_pred, rel_root_depth_pred, hand_type_pred, targets, meta_info):
        target_joint_heatmap = render_gaussian_heatmap(targets['joint_coord'])
        
        loss = {}
        loss['joint_heatmap'] = self.joint_heatmap_loss(joint_heatmap_pred, target_joint_heatmap, meta_info['joint_valid'])
        loss['rel_root_depth'] = self.rel_root_depth_loss(rel_root_depth_pred, targets['rel_root_depth'], meta_info['root_valid'])
        loss['hand_type'] = self.hand_type_loss(hand_type_pred, targets['hand_type'], meta_info['hand_type_valid'])

    
        return loss