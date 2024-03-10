import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from collections import OrderedDict


from config import config as cfg
from dataloader.InterHand2M6.dataset import Dataset as InterHand2M6Dataset
from dataloader.RHD.dataset import Dataset as RHDDataset
from dataloader.STB.dataset import Dataset as STBDataset

from common.timer import Timer
from common.logger import colorlogger
from common.interNet import InterNet

cfg.is_inference = False

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
            if cfg.gpu_idx is None:
                device = torch.device(f"cuda")
            else:
                device = torch.device(f"cuda:{gpu_index}")
            cudnn.benchmark = True
        else:
            print("CUDA is unavailable, using CPU")
            device = torch.device("cpu")
        
        self.dataset = cfg.dataset
        self.logger.info("Creating train dataset...")
        if self.dataset == 'InterHand2.6M':
            train_set = InterHand2M6Dataset(transforms.ToTensor(), "train")
            val_set = InterHand2M6Dataset(transforms.ToTensor(), "val")
        elif self.dataset == 'RHD':
            train_set = RHDDataset(transforms.ToTensor(), "train")
            val_set = RHDDataset(transforms.ToTensor(), "val")
        elif self.dataset == 'STB':
            train_set = STBDataset(transforms.ToTensor(), "train")
            val_set = STBDataset(transforms.ToTensor(), "val")
        else:
            raise ValueError('Invalid dataset name')
        
        joint_num = train_set.joint_num

        self.device = device
        self.logger.info("Creating graph and optimizer...")
        model = InterNet(joint_num)
        # model = DataParallel(model).cuda()
        
        self.model = model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)


        if cfg.fast_debug:
            train_batch_size = 2
            test_batch_size = 2
        else:
            train_batch_size = cfg.train_batch_size
            test_batch_size = cfg.test_batch_size
                    
        self.train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        
        current_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


        # log_dir_last_name = sorted(glob.glob(os.path.join(save_log_dir, dataset_name, 'run_*')), key=lambda x: int(x.split('_')[-1]))
        # run_id = int(log_dir_last_name[-1].split('_')[-1]) + 1 if log_dir_last_name else 0

        self.exp_dir = os.path.join(cfg.save_log_dir, cfg.model_name, cfg.dataset, f'run_{current_timestamp}')
        os.makedirs(self.exp_dir, exist_ok=True)

        self.txtfile = os.path.join(self.exp_dir, 'log.txt')
        if os.path.exists(self.txtfile):
            os.remove(self.txtfile)

        self.write_loginfo_to_txt(f'{self.exp_dir}')

        self.logger = SummaryWriter(self.exp_dir)

        self.best_val_epoch = float('inf')
        self.start_epoch = 0
        finetune = cfg.fine_tune

        if cfg.resume_weight_path is not None:
            if not os.path.isfile(cfg.resume_weight_path):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(cfg.resume_weight_path))
            checkpoint = torch.load(cfg.resume_weight_path, map_location=torch.device('cpu'))
            # model.load_state_dict(checkpoint["state_dict"])
            # Using the following load method will cause each process to occupy an extra part of the video memory on GPU0. The reason is that the default load location is GPU0.
            # checkpoint = torch.load("checkpoint.pth")

            # Update the model's state dict
            
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
                self.model.load_state_dict(new_state_dict, strict=False)
            else:
                new_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in self.model.state_dict()}
                self.model.load_state_dict(new_state_dict, strict=False)

           
            # if cuda_valid:
            #     self.model.module.load_state_dict(checkpoint['state_dict'])
            # else:
            #     self.model.load_state_dict(checkpoint['state_dict'])
            
            # Check if the models are different
            old_keys = set(checkpoint['state_dict'].keys())
            new_keys = set(self.model.state_dict().keys())

            # If there's a difference in the keys, we assume the architectures are different
            if old_keys != new_keys:
                finetune = True
            else:
                finetune = False  # or set finetune based on some other condition or user input

            # Conditional loading of the optimizer state
            if not finetune: #train                
                self.best_val_epoch_mpjpe = checkpoint['MPJPE']
                self.start_epoch = checkpoint['epoch']

                # However, if you do want to load the state dict, you would need to ensure that the state matches the new model
                optimizer_state_dict = checkpoint['optimizer']
                
                # Filter out optimizer state that doesn't match the new model's parameters
                filtered_optimizer_state_dict = {
                    k: v for k, v in optimizer_state_dict.items() if k in self.optimizer.state_dict()
                }
                
                # Load the filtered state dict
                self.optimizer.load_state_dict(filtered_optimizer_state_dict)


            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.resume_weight_path, checkpoint['epoch']))
            self.write_loginfo_to_txt("=> loaded checkpoint '{}' (epoch {})".format(cfg.resume_weight_path, checkpoint['epoch'])+'\n\n')
            # Clear start epoch if fine-tuning
            if finetune:
                self.start_epoch = 0

        self.model.to(device)
        shutil.copy('config/config.py', f'{self.exp_dir}/config.py')

        
    def training(self, cur_epoch, total_epoch, split):
        self.model.train()
        self.set_lr(cur_epoch)
        tbar = tqdm(self.train_loader)
        num_iter = len(self.train_loader)

        width = 10  # Total width including the string length
        formatted_split = split.rjust(width)
        epoch_loss = []

        for idx, (inputs, targets, meta_info) in enumerate(tbar): # 6 ~ 10 s
            if cfg.fast_debug and idx > 2:
                break     
            self.optimizer.zero_grad()
            img = inputs['img'].to(self.device)
            
            for key, value in meta_info.items():
                meta_info[key] = value.to(self.device)
            for key, value in targets.items():
                targets[key] = value.to(self.device)
                
            joint_heatmap_pred, rel_root_depth_pred, hand_type_pred = self.model(img)
            loss = self.model.compute_loss(joint_heatmap_pred, rel_root_depth_pred, hand_type_pred, targets, meta_info)

            joint_heatmap_loss = loss['joint_heatmap']
            rel_root_depth_loss = loss['rel_root_depth']
            hand_type_loss = loss['hand_type']
            loss = joint_heatmap_loss + rel_root_depth_loss + hand_type_loss 
            loss.backward()
            self.optimizer.step()

            loginfo = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Iter: {idx:05d}/{num_iter:05d}, Loss: {loss.item():.4f}'
            loginfo += f'| L_jh: {joint_heatmap_loss.item():.4f}'
            loginfo += f'| L_rrd: {rel_root_depth_loss.item():.4f}'
            loginfo += f'| L_ht: {hand_type_loss.item():.4f}'

            tbar.set_description(loginfo)

            iter_loss_value = round(loss.item(), 4)
            epoch_loss.append(iter_loss_value)

        epoch_loss_value = np.round(np.mean(epoch_loss), 4)
        self.logger.add_scalar(f'{formatted_split} epoch MPJPE', epoch_loss_value, global_step=cur_epoch)
        epoch_info = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Loss: {epoch_loss_value}'            
        # epoch_mpjpe = np.round(np.mean(epoch_mpjpe), 5)
        print(epoch_info)
        self.write_loginfo_to_txt(epoch_info)
        return epoch_loss_value
    
    
    def validation(self, cur_epoch, total_epoch, split):
        self.model.eval()
        tbar = tqdm(self.val_loader)
        num_iter = len(self.val_loader)

        width = 10  # Total width including the string length
        formatted_split = split.rjust(width)
        epoch_loss = []
        epoch_mpjpe = []

        for idx, (inputs, targets, meta_info) in enumerate(tbar): # 6 ~ 10 s
            if cfg.fast_debug and idx > 2:
                break     
            self.optimizer.zero_grad()
            with torch.no_grad():
                img = inputs['img'].to(self.device)
            
                for key, value in meta_info.items():
                    meta_info[key] = value.to(self.device)
                for key, value in targets.items():
                    targets[key] = value.to(self.device)
                    
                joint_heatmap_pred, rel_root_depth_pred, hand_type_pred = self.model(img)
                
                loss = self.model.compute_loss(joint_heatmap_pred, rel_root_depth_pred, hand_type_pred, targets, meta_info)

            joint_heatmap_loss = loss['joint_heatmap']
            rel_root_depth_loss = loss['rel_root_depth']
            hand_type_loss = loss['hand_type']
            loss = joint_heatmap_loss + rel_root_depth_loss + hand_type_loss 

            loginfo = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Iter: {idx:05d}/{num_iter:05d}, Loss: {loss.item():.4f}'
            loginfo += f'| L_jh: {joint_heatmap_loss.item():.4f}'
            loginfo += f'| L_rrd: {rel_root_depth_loss.item():.4f}'
            loginfo += f'| L_ht: {hand_type_loss.item():.4f}'

            tbar.set_description(loginfo)

            iter_loss_value = round(loss.item(), 4)
            epoch_loss.append(iter_loss_value)

        epoch_loss_value = np.round(np.mean(epoch_loss), 4)
        self.logger.add_scalar(f'{formatted_split} epoch MPJPE', epoch_loss_value, global_step=cur_epoch)
        epoch_info = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Loss: {epoch_loss_value}'            
        # epoch_mpjpe = np.round(np.mean(epoch_mpjpe), 5)
        
        self.write_loginfo_to_txt(epoch_info)
        self.write_loginfo_to_txt('')
        print(epoch_info)
        # print('')
        return epoch_loss_value
    

    def save_checkpoint(self, state, is_best, ouput_weight_dir = ''):
        """Saves checkpoint to disk"""
        os.makedirs(ouput_weight_dir, exist_ok=True)
        best_model_filepath = os.path.join(ouput_weight_dir, f'model_best.pth.tar')
        filename = os.path.join(ouput_weight_dir, f'checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:   
            torch.save(state, best_model_filepath)
    
    def write_loginfo_to_txt(self, loginfo):
        loss_file = open(self.txtfile, "a+")
        if loginfo.endswith('\n'):
            loss_file.write(loginfo)
        else:
            loss_file.write(loginfo+'\n')
        loss_file.close()# 
    
    def set_lr(self, epoch):
        if len(cfg.lr_dec_epoch) == 0:
            return cfg.lr

        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr

    def run(self):
        for epoch in range(self.start_epoch, cfg.end_epoch): 
            # _ = self.trainval(epoch, max_epoch, self.val_loader, 'training', fast_debug = fast_debug)
            self.training(epoch, cfg.end_epoch, self.train_loader, 'training')

            epoch_loss = self.validation(epoch, cfg.end_epoch, self.val_loader, 'validation')
            checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch_loss': epoch_loss,                
                        }
            if epoch_loss < self.best_val_epoch:
                self.best_val_epoch = epoch_loss
                is_best = True
            else:
                is_best = False

            self.save_checkpoint(checkpoint, is_best, self.exp_dir)
            print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--gpuid', type=int, default=0, help='GPU index')
    parser.add_argument('--fast_debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    gpu_idx = args.gpuid
    cfg.fast_debug = args.fast_debug
    # fast_debug = True
    worker = Worker(gpu_idx)
    worker.run()

    # gpu_info = get_gpu_utilization_as_string()
    # print('gpu_info', gpu_info)

# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00