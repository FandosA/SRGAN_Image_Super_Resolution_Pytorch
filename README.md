# Image Super Resolution SRGAN
Unofficial implementation of the SRGAN model based on the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802v5). With this repository you will be able to implement the model from scratch with your own dataset. The steps to train and test the model, as well as the details of the project, are explained below.

## Dataset preparation
To prepare the dataset, you must first place the images with which you will train the model in the ```dataset/``` folder. I used the images from the [TartanAir dataset](https://theairlab.org/tartanair-dataset/) but you can use any images you want. This dataset consists of 16 scenarios with sequences of several images each, the total number of images is 6208. Once the images have been placed in the folder, run:
```
python dataset.py
```
A _json_ file will be created with the paths to the images. This _json_ file will be used in the training to load the images. See the ```dataset/``` folder and the _json_ file in the repository as an example.

## Train the model
To train the model, you can tune the parameters and hyperparameters as you wish. In my case I used the same values ​​as in the paper and trained the model to increase the size of the images by a factor of 2. Regarding the dataset, I used 90% of the images for the training cycle and 10% of the images for the validation cycle. This percentages can also be changed in the parameters. To start the training, run:
```
python train.py
```
When the training starts, a folder is created. The name of this folder consists of the name given in the parameters, plus the relevant training parameter information, such as the batch size, the learning rate, the up scaling factor and the number of residual blocks to add to the network. This way, you always know what parameter values have been used for the training. In my case, the name of my training folder is ```srgan_bs2_lr0.0001_upscalefactor2_numresblocks16/```. Six _txt_ files are stored in this folder. They contain the values of the discriminator and generator loss and the discriminator accuracy at each epoch. Finally, inside this folder, another subfolder, called ```checkpoints/```, is created to store the model every 10 epochs, and also the best model (i.e. the model with the lowest loss value). I provide the best model of the generator I have trained so that you can use it, as well as my _txt_ files with my loss and accuracy values.

## Test the model
To test the model, it is only necessary to set the model to be tested in the parameters and also set the same training parameters used for the training of the model (i.e. upscale factor and number of residual blocks). Then place the desired images to be upscaled in the folder ```test_images/original/``` and run:
```
python test.py
```
The resuting upscaled images will be saved in the folder ```test_images/upscaled/```. As an example, check out the folder with the image I upscaled here in the repository.
