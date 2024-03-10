import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from collections import OrderedDict

import numpy as np
import os
import glob
import shutil
import GPUtil
import time
from datetime import datetime
import platform
import argparse

from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn


from config import config as cfg
from dataloader.InterHand2M6.dataset import Dataset as InterHand2M6Dataset
from dataloader.RHD.dataset import Dataset as RHDDataset
from dataloader.STB.dataset import Dataset as STBDataset

from common.timer import Timer
from common.logger import colorlogger
from common.interNet import InterNet


cfg.is_inference = True

# if platform.system() == 'Windows':
#     print("This is Windows")
# elif platform.system() == 'Linux':
#     print("This is Linux")
# elif platform.system() == 'Darwin':
#     print("This is MacOS")

class Worker(object):
    def __init__(self, gpu_index = None):
        
        cuda_valid = torch.cuda.is_available()
        if cuda_valid:
            gpu_index = gpu_index  # # Here set the index of the GPU you want to use
            print(f"CUDA is available, using GPU {gpu_index}")
            if gpu_idx is None:
                device = torch.device(f"cuda")
            else:
                device = torch.device(f"cuda:{gpu_index}")
            cudnn.benchmark = True
        else:
            print("CUDA is unavailable, using CPU")
            device = torch.device("cpu")
        
        self.dataset = cfg.dataset
        self.split = 'test'
        if self.dataset == 'InterHand2.6M':
            self.data_set = InterHand2M6Dataset(transforms.ToTensor(), self.split)
        elif self.dataset == 'RHD':
            self.data_set = RHDDataset(transforms.ToTensor(), self.split)
        elif self.dataset == 'STB':
            self.data_set = STBDataset(transforms.ToTensor(), self.split)
        else:
            raise ValueError('Invalid dataset name')
        
        joint_num = self.data_set.joint_num

        self.device = device
        # self.logger.info("Creating graph and optimizer...")
        model = InterNet(joint_num, device= self.device)
        # model = DataParallel(model).cuda()
        
        self.model = model.to(device)

        if cfg.fast_debug:
            test_batch_size = 4
        else:
            test_batch_size = cfg.test_batch_size
                    
        self.data_loader = DataLoader(self.data_set, batch_size=test_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        
        save_log_dir = cfg.infer_resume_weight_path[:cfg.infer_resume_weight_path.find(cfg.infer_resume_weight_path.split('/')[-1])]
        log_dir = sorted(glob.glob(os.path.join(save_log_dir, 'infer_*')), key=lambda x: int(x.split('_')[-1]))
        run_id = int(log_dir[-1].split('_')[-1]) + 1 if log_dir else 0
        self.exp_dir = os.path.join(save_log_dir, f'infer_{run_id:03d}')
        os.makedirs(self.exp_dir, exist_ok=True)
        self.save_img = True
        if self.save_img:
            self.img_save_dir = os.path.join(self.exp_dir, 'img')

        if self.save_img:
            os.makedirs(self.img_save_dir, exist_ok=True)

        self.txtfile = os.path.join(self.exp_dir, 'eval_log.txt')
        if os.path.exists(self.txtfile):
            os.remove(self.txtfile)

        # self.logger = SummaryWriter(self.exp_dir)

        
        if not os.path.isfile(cfg.infer_resume_weight_path):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(cfg.infer_resume_weight_path))
        checkpoint = torch.load(cfg.infer_resume_weight_path, map_location=torch.device('cpu'))
        # model.load_state_dict(checkpoint["state_dict"])
        # Using the following load method will cause each process to occupy an extra part of the video memory on GPU0. The reason is that the default load location is GPU0.
        # checkpoint = torch.load("checkpoint.pth")
        # Remove 'module.' prefix
        if 'Pre-trained_weights' in cfg.infer_resume_weight_path:
            new_state_dict = OrderedDict()
            for key, value in checkpoint['network'].items():
                key = key[7:]  # remove `module.` prefix
                # new_state_dict[name] = value
                parts = key.split(".")
                # Assuming the insertion needs to happen after 'resnet'
                if "resnet" in parts:
                    # index = parts.index("resnet") + 1
                    # parts.insert(index, "model")
                    new_key = ".".join(parts)
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            # Load the adjusted state dict
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        
        # if cuda_valid:
        #     self.model.module.load_state_dict(checkpoint['state_dict'])
        # else:
        #     self.model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})".format(cfg.infer_resume_weight_path, checkpoint['epoch']))


    
    def test(self, split = 'test'):
        self.model.eval()
        tbar = tqdm(self.data_loader)
        num_iter = len(self.data_loader)

        width = 10  # Total width including the string length
        formatted_split = split.rjust(width)

        preds = {'joint_coord': [], 'rel_root_depth': [], 'hand_type': [], 'inv_trans': []}

        for idx, (inputs, targets, meta_info) in enumerate(tbar): # 6 ~ 10 s
            if cfg.fast_debug and idx > 1:
                break     
            with torch.no_grad():
                # inputs = {'img': img}
                # targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
                # meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'hand_type_valid': hand_type_valid, 'inv_trans': inv_trans, 'capture': int(data['capture']), 'cam': int(data['cam']), 'frame': int(data['frame'])}
                img = inputs['img'].to(self.device)

                for key, value in meta_info.items():
                    meta_info[key] = value.to(self.device)
                for key, value in targets.items():
                    targets[key] = value.to(self.device)
                # print('img', img.shape)
                joint_heatmap_pred, rel_root_depth_pred, hand_type_pred = self.model(img)

                out = self.model.compute_coordinates(joint_heatmap_pred, rel_root_depth_pred, hand_type_pred, targets, meta_info)
                joint_coord_out = out['joint_coord'].cpu().numpy()
                rel_root_depth_out = out['rel_root_depth'].cpu().numpy()
                hand_type_out = out['hand_type'].cpu().numpy()
                inv_trans = out['inv_trans'].cpu().numpy()

                preds['joint_coord'].append(joint_coord_out)
                preds['rel_root_depth'].append(rel_root_depth_out)
                preds['hand_type'].append(hand_type_out)
                preds['inv_trans'].append(inv_trans)

                loginfo = f'{formatted_split} Iter: {idx:05d}/{num_iter:05d}'

                tbar.set_description(loginfo)

        preds = {k: np.concatenate(v) for k,v in preds.items()}
        self.data_set.evaluate(preds, save_dir = self.img_save_dir)
        
        # self.write_loginfo_to_txt(epoch_info)
        # self.write_loginfo_to_txt('')
        # print(epoch_info)
        # print('')
    

    def write_loginfo_to_txt(self, loginfo):
        loss_file = open(self.txtfile, "a+")
        if loginfo.endswith('\n'):
            loss_file.write(loginfo)
        else:
            loss_file.write(loginfo+'\n')
        loss_file.close()# 
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--gpuid', type=int, default=None, help='GPU index')
    # parser.add_argument('--fast_debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    # cfg.fast_debug = args.fast_debug
    cfg.fast_debug = True
    gpu_idx = args.gpuid
    # fast_debug = True
    worker = Worker(gpu_idx)
    worker.test()

    # gpu_info = get_gpu_utilization_as_string()
    # print('gpu_info', gpu_info)

# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00