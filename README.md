# Image Super Resolution SRGAN
Unofficial implementation of the SRGAN model. With this repository you will be able to implement the model from scratch with your own dataset. The steps to train the model, as well as the details of the project, are explained below.

## Dataset preparation
To prepare the dataset, you must first place the images with which you will train the model in the ```dataset/``` folder. I used the images from the [TartanAir dataset](https://theairlab.org/tartanair-dataset/) but you can use any images you want. Once you have placed the images in the folder, you just have to run the file:
```
python dataset.py
```
A _json_ file will be created with the paths to the images. See the _json_ file in the repository as an example.
