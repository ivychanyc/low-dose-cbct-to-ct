import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import scipy
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import io


class SingleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A') # depend on the dataroot name make dataset /MR2CT/testA
        print('I am here 14')
        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths, key=lambda x: int(x.split('_')[-1].split('.')[-2]))  # sort as increasing of i : 1,2,3...

        #self.transform = get_transform(opt)
        self.transform = None 

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        # A_img = Image.open(A_path).convert('RGB')
        # A = self.transform(A_img)
        A_img = Image.open(A_path)
        print(A_path)
        value1=65536/2
        value2=65536/2
       
        A_nparray=(np.asarray(A_img)).astype(np.float32)
        A_nparray=scipy.ndimage.zoom(A_nparray,0.5,order=1)
        A_norm=(A_nparray-value1)/value2
        A_tensor=torch.from_numpy(A_norm).unsqueeze(0)
        A=A_tensor.repeat(3,1,1)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
