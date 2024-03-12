# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import torch.utils.data
import cv2
import os
import os.path as osp
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import sys, os
import shutil

sys.path.append('../..')
from config import config as cfg
from utils.preprocessing import load_img, load_skeleton, process_bbox, get_aug_config, augmentation, transform_input_to_output_space, generate_patch_image, trans_point2d
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints
from utils.compute_heatmap import render_gaussian_heatmap

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode):
        # assert mode in ['training', 'evaluation']
        self.mode = mode
        self.root_path = '/scratch/rhong5/dataset/RHD'
        self.rootnet_output_path = f'{self.root_path}/rootnet_output/rootnet_rhd_output.json'
        self.original_img_shape = (320, 320) # height, width
        self.transform = transform
        self.joint_num = 21 # single hand
        self.joint_type = {'right': np.arange(self.joint_num,self.joint_num*2), 'left': np.arange(0,self.joint_num)}
        self.root_joint_idx = {'right': 21, 'left': 0}
        self.skeleton = load_skeleton(osp.join(self.root_path, 'skeleton.txt'), self.joint_num*2)
        
        self.datalist = []
        if self.mode == 'train':
            set = 'training'
        else:
            set = 'evaluation'
        self.annot_path = osp.join(self.root_path, 'RHD_' + set + '.json')
        db = COCO(self.annot_path)
       
        if self.mode == 'test' and cfg.trans_test == 'rootnet':
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")

        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            
            img_path = osp.join(self.root_path, set, 'color', img['file_name'])
            img_width, img_height = img['width'], img['height']
            cam_param = img['cam_param']
            focal, princpt = np.array(cam_param['focal'],dtype=np.float32), np.array(cam_param['princpt'],dtype=np.float32)
            
            joint_img = np.array(ann['joint_img'],dtype=np.float32)
            joint_cam = np.array(ann['joint_cam'],dtype=np.float32)
            joint_valid = np.array(ann['joint_valid'],dtype=np.float32)
            
            # transform single hand data to double hand data structure
            hand_type = ann['hand_type']
            joint_img_dh = np.zeros((self.joint_num*2,2),dtype=np.float32)
            joint_cam_dh = np.zeros((self.joint_num*2,3),dtype=np.float32)
            joint_valid_dh = np.zeros((self.joint_num*2),dtype=np.float32)
            joint_img_dh[self.joint_type[hand_type]] = joint_img
            joint_cam_dh[self.joint_type[hand_type]] = joint_cam
            joint_valid_dh[self.joint_type[hand_type]] = joint_valid
            
            joint_img = joint_img_dh
            joint_cam = joint_cam_dh
            joint_valid = joint_valid_dh

            if self.mode == 'test' and cfg.trans_test == 'rootnet':
                bbox = np.array(rootnet_result[str(aid)]['bbox'],dtype=np.float32)
                abs_depth = rootnet_result[str(aid)]['abs_depth']
            else:
                bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))
                abs_depth = joint_cam[self.root_joint_idx[hand_type],2] # single hand abs depth

            cam_param = {'focal': focal, 
                         'princpt': princpt}
            
            joint = {'cam_coord': joint_cam, 
                     'img_coord': joint_img, 
                     'valid': joint_valid}

            data = {'img_path': img_path, 
                    'bbox': bbox, 
                    'cam_param': cam_param, 
                    'joint': joint, 
                    'hand_type': hand_type, 
                    'abs_depth': abs_depth}
            self.datalist.append(data)
     
    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
  
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        '''
        Even though the joint_can and joint_img are 42-dim, only one between the first 21-dim and the second 21-dim is valid, which means
        either a left hand or a right hand is valid at a time.
        '''
        data = self.datalist[idx]
        img_path = data['img_path']
        bbox = data['bbox'] # x,y,w,h, main hand bbox
        joint = data['joint']
        hand_type = data['hand_type'] ## right or left
        joint_cam = joint['cam_coord'].copy() ## joint xyz coordinates in camera coordinate system, 3-dim vector with a length of 42, 0~20 for left hand, 21~41 for right hand
        joint_img = joint['img_coord'].copy() ## joint uv coordinates in pixel coordinate system
        joint_valid = joint['valid'].copy() ## joint valid flags, 0 for invalid joint and 1 for valid joint, 1-dim vector with a length of 42

        # print('img_path', img_path)
        # print('hand_type', hand_type)
        # print('joint_cam\n', joint_cam)
        # print('joint_img\n', joint_img)
        # print('bbox\n', bbox)
        # print('joint_valid\n', joint_valid)
        # # print('joint_valid', joint_valid.shape) # (42,)
        hand_type = self.handtype_str2array(hand_type)
        # print('hand_type', hand_type) # [1. 0.] for right or [0. 1.] for left
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)
        # print('joint_coord\n', joint_coord) # (42, 3), uvz, 0~20 for left hand, 21~41 for right hand
        

        # image load
        img = load_img(img_path)
        # augmentation
        crop_img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, 
                                                                                hand_type, self.mode, self.joint_type)
        print('-----------joint_coord\n', joint_coord)
        print('joint_valid\n', joint_valid)
        print('hand_type\n', hand_type)
        print('inv_trans\n', inv_trans)

        
        # crop hand region from whole image
        crop_img = self.transform(crop_img.astype(np.float32))/255.
        rel_root_depth = np.zeros((1),dtype=np.float32)
        root_valid = np.zeros((1),dtype=np.float32)
        
        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, 
                                                                                               root_valid, self.root_joint_idx, self.joint_type)
        ## joint_coord in output heatmap space

        # print('joint_coord', joint_coord.shape) # (42, 3)
        # print('rel_root_depth', rel_root_depth.shape) # (1,)
        # print('hand_type', hand_type.shape) # (2,)
        # print('22222222222 joint_coord\n', joint_coord) ## joint_coord in output heatmap space
        # print('joint_valid\n', joint_valid)
        # print('rel_root_depth\n', rel_root_depth) ## all rel_root_depth are 32
        # print('root_valid\n', root_valid) ## all root_valid are 0

        inputs = {'img': crop_img, 'img_path': img_path}
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'inv_trans': inv_trans, 'hand_type_valid': 1}

        return inputs, targets, meta_info, data
    

    def evaluate_per_iter(self, preds, targets = None, meta_info = None, mode = 'test',  save_dir = None):
        pass
    
    def evaluate(self, preds, mode = 'test',  save_dir = None):
        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord = preds['joint_coord']
        preds_rel_root_depth = preds['rel_root_depth']
        preds_hand_type = preds['hand_type']
        inv_trans =  preds['inv_trans']
                # if preds_joint_coord is tensor, convert it to numpy
        if isinstance(preds_joint_coord, torch.Tensor):
            preds_joint_coord = preds_joint_coord.cpu().numpy()
            preds_rel_root_depth = preds_rel_root_depth.cpu().numpy()
            preds_hand_type = preds_hand_type.cpu().numpy()
            inv_trans = inv_trans.cpu().numpy()

        # assert len(gts) == len(preds_joint_coord)
        # sample_num = len(gts)
        sample_num = preds_joint_coord.shape[0]
        
        mpjpe = [[] for _ in range(self.joint_num)] # treat right and left hand identical
        acc_hand_cls = 0
        for n in range(sample_num):
            data = gts[n]

            bbox = data['bbox']
            cam_param = data['cam_param']
            joint = data['joint']
            gt_hand_type = data['hand_type']

            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord']
            joint_valid = joint['valid']

            # restore coordinates to original space
            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
            pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
            for j in range(self.joint_num*2):
                pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])
            pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)
            pred_joint_coord_img[:,2] = pred_joint_coord_img[:,2] + data['abs_depth']

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

            # root joint alignment, convert absolute coordinates to the relative coordinates
            pred_joint_coord_cam[self.joint_type['right']] = pred_joint_coord_cam[self.joint_type['right']] - pred_joint_coord_cam[self.root_joint_idx['right'],None,:]
            pred_joint_coord_cam[self.joint_type['left']] = pred_joint_coord_cam[self.joint_type['left']] - pred_joint_coord_cam[self.root_joint_idx['left'],None,:]
            gt_joint_coord[self.joint_type['right']] = gt_joint_coord[self.joint_type['right']] - gt_joint_coord[self.root_joint_idx['right'],None,:]
            gt_joint_coord[self.joint_type['left']] = gt_joint_coord[self.joint_type['left']] - gt_joint_coord[self.root_joint_idx['left'],None,:]
            
            # select right or left hand using groundtruth hand type
            pred_joint_coord_cam = pred_joint_coord_cam[self.joint_type[gt_hand_type]]
            gt_joint_coord = gt_joint_coord[self.joint_type[gt_hand_type]]
            joint_valid = joint_valid[self.joint_type[gt_hand_type]]

            # mpjpe save
            for j in range(self.joint_num):
                if joint_valid[j]:
                    mpjpe[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
            
            if gt_hand_type == 'right' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] < 0.5:
                acc_hand_cls += 1
            elif gt_hand_type == 'left' and preds_hand_type[n][0] < 0.5 and preds_hand_type[n][1] > 0.5:
                acc_hand_cls += 1

            vis = True
            if not save_dir is None and vis:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                vis_kps = pred_joint_coord_img.copy()
                vis_valid = joint_valid.copy()
                filename = 'out_' + str(n) + '.jpg'
                vis_keypoints(_img, vis_kps, vis_valid, self.skeleton[:self.joint_num], filename, save_path = save_dir)

            vis = True
            if not save_dir is None and vis:
                filename = 'out_' + str(n) + '_3d.png'
                vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton[:self.joint_num], filename, save_path = save_dir)

            

        print('Handedness accuracy: ' + str(acc_hand_cls / sample_num))

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num):
            if len(mpjpe[j]) > 0:
                mpjpe[j] = np.mean(np.stack(mpjpe[j]))
                joint_name = self.skeleton[j]['name']
                eval_summary += (joint_name + ': %.2f, ' % mpjpe[j])
        print(eval_summary)
        print('MPJPE: %.2f' % (np.mean(mpjpe)))



if __name__ == '__main__':
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Dataset(transform, 'val')
    batch_size = 1
    num_workers = 0
    # Creating the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers)


    i = 0
    for batch in dataloader:

        inputs, targets, meta_info, data = batch
        '''        inputs = {'img': img}
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'inv_trans': inv_trans, 'hand_type_valid': 1}
        '''
        # print(inputs['img'].shape) # torch.Size([bs, 3, 256, 256])
        # print(targets['joint_coord'].shape) # torch.Size([bs, 42, 3])
        # print(targets['rel_root_depth'].shape) # torch.Size([bs, 1])
        # print(targets['hand_type'].shape) # torch.Size([bs, 2])
        # print(meta_info['joint_valid'].shape) # torch.Size([bs, 42])
        # print(meta_info['root_valid'].shape) # torch.Size([bs, 1])
        # print(meta_info['inv_trans'].shape) # torch.Size([bs, 2, 3])
        # print(meta_info['hand_type_valid'].shape) # torch.Size([bs, 1])


        # img = (inputs['img'].cpu().numpy()*255).astype(np.uint8)
        img_path = inputs['img_path']
        # img_name = img_path[0].split('/')[-1]
        # shutil.copy(img_path[0], f'img_examples/{img_name}')
        # cv2.imwrite(f"img_examples/{img_name.split('.')[0]}_crop.{img_name.split('.')[1]}", img[0].transpose(1,2,0)[:,:,::-1])

        # gaussian_heatmap = render_gaussian_heatmap(targets['joint_coord'])
        # gaussian_heatmap = gaussian_heatmap.cpu().numpy().astype(np.uint8)[0]
        # print(gaussian_heatmap.shape)


        # break
        i += 1
        print('')
        if i > 5:
            break