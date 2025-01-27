# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:26:05 2025

@author: fandos
"""

import os
import torch
import numpy as np
import configargparse
import torch.nn as nn
from loss import GeneratorLoss
from dataset import Dataset
from matplotlib import pyplot as plt
from srgan_model import Generator, Discriminator
from utils import selectDevice, plotLoss, calculateAccuracy, plotAccuracy


def train():
    
    min_val_loss_g = np.inf
    min_val_loss_d = np.inf
    bestEpoch_g = 0
    bestEpoch_d = 0

    train_losses_g = []
    val_losses_g = []
    
    train_accuracies_d = []
    val_accuracies_d = []
    train_losses_d = []
    val_losses_d = []
    
    
    print('--------------------------------------------------------------')
    
    # Loop along epochs to do the training
    for i in range(args.epochs + 1):
        
        print(f'EPOCH {i}')
        
        # Training loop
        train_loss_g = 0.0
        train_acc_d = 0.0
        train_loss_d = 0.0
        generator.train()
        discriminator.train()
        iteration = 1
        
        print('\nTRAINING')
        
        for image_hr, image_lr in train_loader:
            
            print('\rEpoch[' + str(i) + '/' + str(args.epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(train_loader)), end='')
            iteration += 1
            
            image_hr, image_lr = image_hr.to(device), image_lr.to(device)
            
            """
            ###########################################
            #              Discriminator              #
            ###########################################
            """
            optimizer_d.zero_grad()

            image_fake = generator(image_lr).detach()
            
            real_logits = discriminator(image_hr)
            fake_logits = discriminator(image_fake)
            
            real_loss = adversarial_loss(real_logits, torch.ones_like(real_logits))
            fake_loss = adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
            loss_d = (real_loss + fake_loss) / 2

            loss_d.backward()
            optimizer_d.step()
            
            real_accuracy = calculateAccuracy(real_logits, torch.ones_like(real_logits))
            fake_accuracy = calculateAccuracy(fake_logits, torch.zeros_like(fake_logits))
            train_acc_d += (real_accuracy + fake_accuracy).item() / 2
            train_loss_d += loss_d.item()
            
            
            """
            ###########################################
            #                Generator                #
            ###########################################
            """
            optimizer_g.zero_grad()
            
            image_fake = generator(image_lr)
            fake_logits = discriminator(image_fake)
            
            loss_g = generator_loss(image_fake, image_hr, fake_logits)
            
            loss_g.backward()
            optimizer_g.step()
            
            train_loss_g += loss_g.item()
        
        
        # Validation loop
        val_loss_g = 0.0
        val_acc_d = 0.0
        val_loss_d = 0.0
        generator.eval()
        discriminator.eval()
        iteration = 1

        print('')
        print('\nVALIDATION')
        
        with torch.no_grad():
            
            for image_hr, image_lr in validate_loader:
                
                print('\rEpoch[' + str(i) + '/' + str(args.epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(validate_loader)), end='')
                iteration += 1
                
                image_hr, image_lr = image_hr.to(device), image_lr.to(device)
                
                """
                ###########################################
                #              Discriminator              #
                ###########################################
                """
                image_fake = generator(image_lr)
        
                real_logits = discriminator(image_hr)
                fake_logits = discriminator(image_fake)
                
                real_loss = adversarial_loss(real_logits, torch.ones_like(real_logits))
                fake_loss = adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
                loss_d = (real_loss + fake_loss) / 2
                
                real_accuracy = calculateAccuracy(real_logits, torch.ones_like(real_logits))
                fake_accuracy = calculateAccuracy(fake_logits, torch.zeros_like(fake_logits))
                val_acc_d += (real_accuracy + fake_accuracy).item() / 2
                val_loss_d += loss_d.item()
                
                
                """
                ###########################################
                #                Generator                #
                ###########################################
                """
                loss_g = generator_loss(image_fake, image_hr, fake_logits)
                val_loss_g += loss_g.item()
        

        scheduler_g.step()
        scheduler_d.step()

        # Save loss and accuracy values
        train_accuracies_d.append(train_acc_d / len(train_loader))
        val_accuracies_d.append(val_acc_d / len(validate_loader))
        train_losses_d.append(train_loss_d / len(train_loader))
        val_losses_d.append(val_loss_d / len(validate_loader))
        
        train_losses_g.append(train_loss_g / len(train_loader))
        val_losses_g.append(val_loss_g / len(validate_loader))
        
        print("\n\nDiscriminator")
        print(f'- Train accuracy: {train_acc_d / len(train_loader):.3f}')
        print(f'- Validation accuracy: {val_acc_d / len(validate_loader):.3f}')
        print(f'- Train loss: {train_loss_d / len(train_loader):.3f}')
        print(f'- Validation loss: {val_loss_d / len(validate_loader):.3f}\n')
        print("Generator")
        print(f'- Train loss G: {train_loss_g / len(train_loader):.3f}')
        print(f'- Validation loss G: {val_loss_g / len(validate_loader):.3f}')
        
        # Save the model every 10 epochs
        if i % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_g.pth"))
            torch.save(discriminator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_d.pth"))
            
        # Save the best generator model when loss decreases respect to the previous best loss
        if (val_loss_g / len(validate_loader)) < min_val_loss_g:
            # If first epoch, save model as best, otherwise, replace the previous best model with the current one
            if i == 0:
                torch.save(generator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best_g.pth"))
            else:
                os.remove(os.path.join(checkpoints_path, "checkpoint_" + str(bestEpoch_g) + "_best_g.pth"))
                torch.save(generator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best_g.pth"))
            
            print(f'\nValidation loss of Generator decreased: {min_val_loss_g:.3f} ---> {val_loss_g / len(validate_loader):.3f}\nModel saved')
                
            # Update parameters with the new best model
            min_val_loss_g = val_loss_g / len(validate_loader)
            bestEpoch_g = i
            
        # Save the best discriminator model when loss decreases respect to the previous best loss
        if (val_loss_d / len(validate_loader)) < min_val_loss_d:
            # If first epoch, save model as best, otherwise, replace the previous best model with the current one
            if i == 0:
                torch.save(discriminator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best_d.pth"))
            else:
                os.remove(os.path.join(checkpoints_path, "checkpoint_" + str(bestEpoch_d) + "_best_d.pth"))
                torch.save(discriminator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best_d.pth"))
            
            print(f'\nValidation loss of Discriminator decreased: {min_val_loss_d:.3f} ---> {val_loss_d / len(validate_loader):.3f}\nModel saved')
                
            # Update parameters with the new best model
            min_val_loss_d = val_loss_d / len(validate_loader)
            bestEpoch_d = i
            
        np.savetxt(os.path.join(log_dir_path, 'train_losses_g.txt'), np.array(train_losses_g))
        np.savetxt(os.path.join(log_dir_path, 'val_losses_g.txt'), np.array(val_losses_g))
        np.savetxt(os.path.join(log_dir_path, 'train_losses_d.txt'), np.array(train_losses_d))
        np.savetxt(os.path.join(log_dir_path, 'val_losses_d.txt'), np.array(val_losses_d))
        np.savetxt(os.path.join(log_dir_path, 'train_accs_d.txt'), np.array(train_accuracies_d))
        np.savetxt(os.path.join(log_dir_path, 'val_accs_d.txt'), np.array(val_accuracies_d))
            
        print("--------------------------------------------------------------")
    
    # Plot loss and accuracy curves
    plotLoss(log_dir_path, np.array(train_losses_g), np.array(val_losses_g), "loss_generator")
    plotLoss(log_dir_path, np.array(train_losses_d), np.array(val_losses_d), "loss_discriminator")
    plotAccuracy(log_dir_path, np.array(train_accuracies_d), np.array(val_accuracies_d), "accuracy_discriminator")
    plt.show()


if __name__ == "__main__":
    
    # Select parameters for training
    arg = configargparse.ArgumentParser()
    arg.add_argument('--dataset_file_path', type=str, default='images_paths.json', help='Dataset path.')
    arg.add_argument('--train_split', type=float, default=0.9, help='Percentage of the dataset to use for training.')
    arg.add_argument('--log_dir', type=str, default='srgan', help='Name of the folder to save the model.')
    arg.add_argument('--batch_size', type=int, default=2, help='Batch size.')
    arg.add_argument('--upscale_factor', type=int, default=2, help='Upscale factor.')
    arg.add_argument('--num_workers', type=int, default=4, help='Number of threads to use in order to load the dataset.')
    arg.add_argument('--num_resblocks', type=int, default=16, help='Number of residual blocks for the generator.')
    arg.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    arg.add_argument('--epochs', type=int, default=400, help='Number of epochs.')
    arg.add_argument('--GPU', type=bool, default=True, help='True to train the model in the GPU.')
    args = arg.parse_args()
    
    assert (args.train_split < 1), 'The percentage of the dataset to use for training must be lower than 1'
    assert (args.upscale_factor == 2 or args.upscale_factor == 4 or args.upscale_factor == 8), 'The upscale factor only can be 2, 4 or 8'
    
    log_dir_path = args.log_dir + "_bs" + str(args.batch_size) + "_lr" + str(args.learning_rate) + "_upscalefactor" + str(args.upscale_factor) + "_numresblocks" + str(args.num_resblocks)
    assert not (os.path.isdir(log_dir_path)), 'The folder log_dir already exists, remove it or change its name'
    
    # Create folder to store checkpoints and training and validation losses and accuracies
    os.mkdir(log_dir_path)
    checkpoints_path = os.path.join(log_dir_path, 'checkpoints')
    os.mkdir(checkpoints_path)
    
    # Select device
    device = selectDevice(args)
            
    # Load dataset and dataloaders
    dataset = Dataset(args, device)
    train_loader, validate_loader = dataset.loadDataloaders()
    
    # Create models
    generator = Generator(n_resblocks=args.num_resblocks, upscale_factor=args.upscale_factor).to(device)
    discriminator = Discriminator().to(device)
    
    # Set up optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)#, weight_decay=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)#, weight_decay=1e-4)
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.1)
    
    # Create the loss functions
    generator_loss = GeneratorLoss(device)
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    # Train the model
    train()