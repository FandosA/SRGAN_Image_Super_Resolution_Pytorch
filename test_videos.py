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
    
    # Select parameters for testing
    arg = configargparse.ArgumentParser()
    arg.add_argument('--dataset_path', type=str, default='test_videos', help='Dataset path.')
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
    
    video_paths = []
    for root, _, files in os.walk(os.path.join(args.dataset_path, "original")):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(root, file))
    
    with torch.no_grad():
        
        for video_path in video_paths:
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
            # Leer un frame para determinar tamaño
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading first frame from video: {video_path}")
                continue
            
            tensor_image = torch.unsqueeze(imageToTensor(frame), dim=0).to(device)
            
            out_image = generator(tensor_image)
            out_image = tensorToImage(out_image)
            
            height, width = out_image.shape[:2]

            # Reset video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
            output_dir = os.path.join(args.dataset_path, "upscaled")
            output_name = os.path.splitext(os.path.basename(video_path))[0] + "_upscaled.mp4"
            output_path = os.path.join(output_dir, output_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for _ in range(total_frames):
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                tensor = torch.unsqueeze(imageToTensor(frame), dim=0).to(device)
                out_tensor = generator(tensor)
                out_image = tensorToImage(out_tensor)
                out_video.write(out_image)
                
                cv2.imshow("Original", frame)
                cv2.imshow("Upscaled", out_image)
                
                key = cv2.waitKey(1)
                if key == 27:
                    break

            cap.release()
            out_video.release()
            cv2.destroyAllWindows()