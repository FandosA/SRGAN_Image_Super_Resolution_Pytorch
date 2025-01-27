# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:26:05 2025

@author: fandos
"""

import os
import cv2
import torch
import configargparse
from srgan_model import Generator
from utils import selectDevice, tensorToImage, imageToTensor


if __name__ == "__main__":
    
    # Select parameters for training
    arg = configargparse.ArgumentParser()
    arg.add_argument('--dataset_path', type=str, default='test_images', help='Dataset path.')
    arg.add_argument('--log_dir', type=str, default='srgan_bs2_lr0.0001_upscalefactor2_numresblocks16', help='Name of the folder where the files of checkpoints and precision and loss values are stored.')
    arg.add_argument('--checkpoint', type=str, default='checkpoint_23_best_g.pth',help='Checkpoint to use')
    arg.add_argument('--upscale_factor', type=int, default=2, help='Upscale factor.')
    arg.add_argument('--num_resblocks', type=int, default=16, help='Number of residual blocks for the generator.')
    arg.add_argument('--GPU', type=bool, default=True, help='True to train the model in the GPU.')
    args = arg.parse_args()
    
    device = selectDevice(args)
    
    generator = Generator(n_resblocks=args.num_resblocks, upscale_factor=args.upscale_factor)
    state_dict = torch.load(os.path.join(args.log_dir, "checkpoints", args.checkpoint), map_location=device)
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    
    image_paths = []

    for root, _, files in os.walk(os.path.join(args.dataset_path, "original")):
        for file in files:
            image_paths.append(os.path.join(root, file))
    
    with torch.no_grad():
        
        for image_test in image_paths:
            
            image_lr = cv2.imread(image_test)
            tensor_image = imageToTensor(image_lr)
            tensor_image = torch.unsqueeze(tensor_image, dim=0).to(device)
            
            image_HR = generator(tensor_image)
            image_HR = tensorToImage(image_HR)
            
            image_name = image_test.split('\\')[-1]
            
            cv2.imwrite(os.path.join(args.dataset_path, "upscaled", image_name), image_HR)
            cv2.imshow('Original Image', image_lr)
            cv2.imshow('High Resolution Image', image_HR)
            cv2.waitKey(0)
            cv2.destroyAllWindows()