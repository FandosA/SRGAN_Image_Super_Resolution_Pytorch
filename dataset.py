# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:18:28 2025

@author: fandos
"""

import os
import cv2
import json
from PIL import Image
import configargparse
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms


class Dataset(Dataset):
    
    def __init__(self, args, device):
        
        self.dataset_path = args.dataset_file_path
        self.device = device
        self.train_split = args.train_split
        self.batch_size = args.batch_size
        self.upscale_factor = args.upscale_factor
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.num_workers = args.num_workers
        
        with open(self.dataset_path, "r") as file:
            self.images_paths = json.load(file)
        
        
    def __len__(self):
        
        return len(self.images_paths)
    
    
    def __getitem__(self, index):
        
        image_hr = cv2.imread(self.images_paths[index])
        image_hr = cv2.cvtColor(image_hr, cv2.COLOR_BGR2RGB)
        
        image_lr = cv2.resize(image_hr, (image_hr.shape[1] // self.upscale_factor, image_hr.shape[0] // self.upscale_factor))

        image_hr = Image.fromarray(image_hr)
        image_lr = Image.fromarray(image_lr)
        
        image_hr = self.transformer(image_hr)
        image_lr = self.transformer(image_lr)
        
        return image_hr, image_lr
    
    
    def loadDataloaders(self):
        
        len_training = int(len(self) * self.train_split)
        len_validation = len(self) - len_training
        
        train_dataset, validate_dataset = random_split(self, [len_training, len_validation])
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        validate_loader = DataLoader(dataset=validate_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        print('Training images: ' + str(len(train_dataset)) + '/' + str(len(self)))
        print('Validation images: ' + str(len(validate_dataset)) + '/' + str(len(self)) + '\n')
        
        return train_loader, validate_loader
    
    
if __name__ == "__main__":
    
    arg = configargparse.ArgumentParser()
    arg.add_argument('--dataset_path', type=str, default='dataset', help='Dataset path.')
    args = arg.parse_args()
    
    image_paths = []

    for root, dirs, files in os.walk(args.dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))

    with open("images_paths.json", 'w') as json_file:
        json.dump(image_paths, json_file, indent=4)
        
    print("Number of images:", str(len(image_paths)))