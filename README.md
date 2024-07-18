# CBCT-to-CT cycleGAN

This PyTorch implementation is adapted from the original cycleGAN open source for the purpose of maping low dose CBCT images into CT images. If you would like to optimize from the original hyperparameters, check out the original cycleGAN at https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

## Getting started

git clone https://gitlab.lrz.de/0000000001355093/cbct-to-ct-cyclegan.git
$ cd CBCT-to-CT cycleGAN

If connected with research server sith SSH, we recommend using MONAI container for this project

$ docker run --gpus all -v /home/your_user_name_here/your_folder_here:/data -ti projectmonai/monai:latest

Or you can also run in the local server, and install PyTorch and other dependencies: 
$ pip install -r requirements.txt.

## Data
Data should be placed in folders in the following path:
Data/trainA
Data/trainB
Data/testA
Data/testB

## Set up Visdom for training monitoring


## Run the train.py





