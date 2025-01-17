# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

model_name = 'InterNet'

## dataset
dataset = 'InterHand2.6M' # InterHand2.6M, RHD, STB
dataset = 'RHD' # InterHand2.6M, RHD, STB

## input, output
input_img_shape = (256, 256)
output_hm_shape = (64, 64, 64) # (depth, height, width)
sigma = 2.5
bbox_3d_size = 400 # depth axis
bbox_3d_size_root = 400 # depth axis
output_root_hm_shape = 64 # depth axis

## model
resnet_type = 50 # 18, 34, 50, 101, 152
joint_num = 21


## training config
lr_dec_epoch = [15, 17] if dataset == 'InterHand2.6M' else [45,47]
end_epoch = 20 if dataset == 'InterHand2.6M' else 50
lr = 1e-4
lr_dec_factor = 10
train_batch_size = 200
val_batch_size = 200

## testing config
test_batch_size = 20
trans_test = 'rootnet' # gt, rootnet


num_workers = 40
## others

save_log_dir = './log'

is_inference = False
fast_debug = False

resume_weight_path = None
fine_tune = False

infer_resume_weight_path = 'Pre-trained_weights/InterHand2.6M/snapshot_20.pth.tar'
# infer_resume_weight_path = 'Pre-trained_weights/RHD/snapshot_49.pth.tar'
infer_resume_weight_path = 'log/InterNet/RHD/run_2024-03-10-23-46-35/model_best.pth.tar'
infer_resume_weight_path = 'log/InterNet/RHD/run_2024-03-11-00-17-27/model_best.pth.tar'

vis_dir = None