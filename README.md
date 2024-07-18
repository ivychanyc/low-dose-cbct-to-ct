# CBCT-to-CT cycleGAN

This PyTorch implementation is adapted from the original cycleGAN open source for the purpose of maping low dose CBCT images into CT images. If you would like to optimize from the original hyperparameters, check out the original cycleGAN at https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

## Getting started

git clone https://gitlab.lrz.de/0000000001355093/cbct-to-ct-cyclegan.git
$ cd CBCT-to-CT cycleGAN

If connected with research server sith SSH, we recommend using MONAI container for this project

$ docker run --gpus all -v /home/your_user_name_here/your_folder_here:/data -ti projectmonai/monai:latest

For example, we can run docker container in the server:
$ docker run --gpus all --shm-size 40G -p 8028:8097 -p 8026:8888 -p 8830:6006 -v /project_data/cbct/low_dose_cbct/cycleGAN/:/data --rm -ti projectmonai/monai:0.1.0

Or you can also run in the local server, and install PyTorch and other dependencies: 
$ pip install -r requirements.txt.

## Data
Data should be placed in folders in the following path:
> data/datasets/trainA
> data/datasets/trainB
> data/datasets/testA
> data/datasets/testB


## Run Visdom server for training monitoring
$ python -m visdom.server

## Train
An example command to run train.py with hyperparameters, 
Note: the container is /data

$ python /data/train.py --dataroot /data/data/datasets/CBCT2CT_subselect_FS_updated/ --name modelCUT_subselectFS_10NCE --gpu_ids 1 

## Test
An example command to run test.py with hyperparameters, 
Note: the container is /data

$ python /data/test.py --dataroot /data/data/datasets/CBCT2CT_subselect_FS_updated/ --name modelCUT_subselectFS_10NCE --gpu_ids 1 --epoch 250 --num_test 5850 --results_dir /data/results_train





