# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:26:05 2025

@author: fandos
"""

import os
import math
import utils
import torch
import numpy as np
import configargparse
import torch.nn as nn
from dataset import Dataset
from srgan_model import Generator, Discriminator


if __name__ == "__main__":
    
    utils.plotAccuracy("srgan_bs2_lr0.0001_upscalefactor2_numresblocks16",
                       np.loadtxt("srgan_bs2_lr0.0001_upscalefactor2_numresblocks16/train_accs_d.txt"),
                       np.loadtxt("srgan_bs2_lr0.0001_upscalefactor2_numresblocks16/val_accs_d.txt"),
                       "discriminator_acc")
    
    utils.plotLoss("srgan_bs2_lr0.0001_upscalefactor2_numresblocks16",
                       np.loadtxt("srgan_bs2_lr0.0001_upscalefactor2_numresblocks16/train_losses_d.txt"),
                       np.loadtxt("srgan_bs2_lr0.0001_upscalefactor2_numresblocks16/val_losses_d.txt"),
                       "discriminator_loss")
    
    utils.plotLoss("srgan_bs2_lr0.0001_upscalefactor2_numresblocks16",
                       np.loadtxt("srgan_bs2_lr0.0001_upscalefactor2_numresblocks16/train_losses_g.txt"),
                       np.loadtxt("srgan_bs2_lr0.0001_upscalefactor2_numresblocks16/val_losses_g.txt"),
                       "generator_loss")