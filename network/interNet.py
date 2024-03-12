# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys, os

sys.path.append('../')

from config import config as cfg
from network.submodules.module import BackboneNet, PoseNet


class InterNet(nn.Module):
    def __init__(self, joint_num = None, device = 'cpu'):
        super(InterNet, self).__init__()
        self.device = device
        if joint_num is None:
            joint_num = cfg.joint_num
        
        self.backbone_net = BackboneNet(self.device)
        self.pose_net = PoseNet(joint_num, device = self.device)

        self.pose_net.apply(init_weights)


    def forward(self, input_img):
        # batch_size = input_img.shape[0]
        img_feat = self.backbone_net(input_img)
        joint_heatmap_out, rel_root_depth_out, hand_type_out = self.pose_net(img_feat)
        # print('joint_heatmap_out', joint_heatmap_out.shape) # torch.Size([bs, 42, 64, 64, 64])
        # print('rel_root_depth_out', rel_root_depth_out.shape) # torch.Size([bs, 1])
        # print('hand_type_out', hand_type_out.shape) # torch.Size([bs, 2])
        
        # self.joint_heatmap_out = joint_heatmap_out
        # self.rel_root_depth_out = rel_root_depth_out
        # self.hand_type = hand_type
        return joint_heatmap_out, rel_root_depth_out, hand_type_out
        
    

    def compute_coordinates(self, joint_heatmap_pred, rel_root_depth_pred, hand_type_pred, targets, meta_info):
        out = {}
        val_z, idx_z = torch.max(joint_heatmap_pred,2)
        val_zy, idx_zy = torch.max(val_z,2)
        val_zyx, joint_x = torch.max(val_zy,2)
        joint_x = joint_x[:,:,None]
        joint_y = torch.gather(idx_zy, 2, joint_x)
        joint_z = torch.gather(idx_z, 2, joint_y[:,:,:,None].repeat(1,1,1,cfg.output_hm_shape[1]))[:,:,0,:]
        joint_z = torch.gather(joint_z, 2, joint_x)
        joint_coord_out = torch.cat((joint_x, joint_y, joint_z),2).float()
        out['joint_coord'] = joint_coord_out
        out['rel_root_depth'] = rel_root_depth_pred
        out['hand_type'] = hand_type_pred
        if 'inv_trans' in meta_info:
            out['inv_trans'] = meta_info['inv_trans']
        if 'joint_coord' in targets:
            out['target_joint'] = targets['joint_coord']
        if 'joint_valid' in meta_info:
            out['joint_valid'] = meta_info['joint_valid']
        if 'hand_type_valid' in meta_info:
            out['hand_type_valid'] = meta_info['hand_type_valid']
        return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)


