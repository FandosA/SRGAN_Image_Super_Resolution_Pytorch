# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:22:22 2025

@author: fandos
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.transforms import Normalize


class ContentLoss(nn.Module):
    
    def __init__(self, device):
        
        super(ContentLoss, self).__init__()
        
        vgg = vgg19(pretrained=True)
        
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:36]).eval().to(device)
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mse_loss = nn.MSELoss()
    
    def forward(self, image_generated, image_real):

        image_generated = (image_generated + 1) / 2
        image_real = (image_real + 1) / 2
        
        image_generated = self.normalize(image_generated)
        image_real = self.normalize(image_real)
        
        gen_features = self.feature_extractor(image_generated)
        target_features = self.feature_extractor(image_real)
        
        loss = self.mse_loss(gen_features, target_features)
        
        return loss
    
    
class TVLoss(nn.Module):
    
    def __init__(self):
        
        super(TVLoss, self).__init__()

    def forward(self, image_generated):
        
        horizontal_diff = torch.abs(image_generated[:, :, 1:, :] - image_generated[:, :, :-1, :])
        vertical_diff = torch.abs(image_generated[:, :, :, 1:] - image_generated[:, :, :, :-1])
        
        tv_loss = torch.sum(horizontal_diff) + torch.sum(vertical_diff)
        
        return tv_loss
    
    
class GeneratorLoss(nn.Module):
    
    def __init__(self, device):
        
        super(GeneratorLoss, self).__init__()
        
        self.content_loss = ContentLoss(device)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, img_generated, img_real, fake_logits):
        
        image_loss = self.mse_loss(img_generated, img_real)
        perceptual_loss = self.content_loss(img_generated, img_real)
        adversarial_loss = self.adversarial_loss(fake_logits, torch.ones_like(fake_logits))
        tv_loss = self.tv_loss(img_generated)
        
        total_loss = perceptual_loss + 0.001 * adversarial_loss + 0.006 * image_loss + 2e-8 * tv_loss
        
        return total_loss