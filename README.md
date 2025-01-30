# Image Super Resolution SRGAN
Unofficial implementation of the SRGAN model based on the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802v5). With this repository, you will be able to implement the model from scratch using your own dataset. The steps to train and test the model, along with the project details, are explained below.

## Dataset preparation
To prepare the dataset, first place the images you will use to train the model in the ```dataset/``` folder. I used images from the [TartanAir dataset](https://theairlab.org/tartanair-dataset/), but you can use any images you prefer. This dataset contains 16 scenarios, each with multiple image sequences, for a total of 6,208 images. Once the images are in the folder, run:
```
python dataset.py
```
A _json_ file will be created with the paths to the images. This _json_ file will be used during training to load the images. Check the ```dataset/``` folder and the _json_ file in the repository for an example.

## Train the model
To train the model, you can tune the parameters and hyperparameters as desired. In my case, I used the same values as in the paper and trained the model to double the size of the images. For the dataset, I used 90% of the images for the training set and 10% for the validation set. These percentages can also be adjusted in the parameters. To start the training, run:
```
python train.py
```
When the training starts, a folder is created. The folder name consists of the name provided in the parameters, followed by relevant training parameter information, such as the batch size, learning rate, upscaling factor, and the number of residual blocks to add to the network. This way, you can always identify the parameter values used for the training. In my case, the name of my training folder is ```srgan_bs2_lr0.0001_upscalefactor2_numresblocks16/```. Six txt files are stored in this folder, containing the values of the discriminator and generator loss, as well as the discriminator accuracy at each epoch. Additionally, within this folder, a subfolder called ```checkpoints/``` is created to store the model every 10 epochs, as well as the best model (i.e., the model with the lowest loss value). I am providing the best generator model I have trained, along with my txt files containing the loss and accuracy values.

## Test the model
To test the model, simply set the model to be tested in the parameters, along with the same training parameters used during the model's training (i.e., upscale factor and number of residual blocks). Then, place the images you want to upscale in the ```test_images/original/``` folder and run:
```
python test.py
```
The resulting upscaled images will be saved in the ```test_images/upscaled/``` folder. As an example, check out the folder containing the image I upscaled, available here in the repository.
