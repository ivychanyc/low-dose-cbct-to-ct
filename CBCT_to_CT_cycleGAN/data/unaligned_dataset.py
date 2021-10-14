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

class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        print(self.dir_A)
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        #print(self.A_paths[0])
        #print(self.maskA_path)
        
        self.A_paths = sorted(self.A_paths)
        self.maskA_paths = self.A_paths
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)
        # self.transform = None #get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        maskA_path = self.maskA_paths[index % self.A_size]
        maskA_path = maskA_path.replace('trainA','maskA')
        #print('****************halo')
        #print(A_path)
        #print(maskA_path)
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        #A_img = Image.open(A_path).convert('RGB') 
        # B_img = Image.open(B_path).convert('RGB')
        
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)
        mask_img = Image.open(maskA_path)
       
        # create Mask vairbale 
        
        value1=65536/2
        value2=65536/2
        # value1=0.0
        # value2=65536.0
       
        A_nparray=(np.asarray(A_img)).astype(np.float32)
        A_nparray=scipy.ndimage.zoom(A_nparray,0.5,order=1)
        A_norm=(A_nparray-value1)/value2

        maskA_nparray=(np.asarray(mask_img)).astype(np.float32)
        maskA_nparray=scipy.ndimage.zoom(maskA_nparray,0.5,order=1)

        # transformed = self.transform(image=image, mask=mask)
        # transformed_image = transformed['image']
        # transformed_mask = transformed['mask']

        if self.transform is not None:
        #    A_norm = self.transform(image=A_norm)["image"]
           transformed = self.transform(image=A_norm, mask=maskA_nparray)
           A_norm = transformed['image']
           maskA_nparray = transformed['mask']

        A_tensor=torch.from_numpy(A_norm).unsqueeze(0)
        A=A_tensor.repeat(3,1,1)

        # if self.transform is not None:
        #    maskA_nparray = self.transform(image=maskA_nparray)["image"]

        maskA_tensor=(torch.from_numpy(maskA_nparray).unsqueeze(0))/65535
        #print('maskA_tensor max', torch.max(maskA_tensor))
        #print('maskA_tensor min', torch.min(maskA_tensor))
        
        # print(maskA_tensor, maskA_tensor.size())

        # print('A_img min max is ',A_img.getextrema())
        # print('A max after np is ', A_nparray.max())
        # print('A min after np is ', A_nparray.min())
        # print('A np shape', A_nparray.shape) 
        # print('A tensor max', torch.max(A))
        # print('A tensor min', torch.min(A))
        # print('tensor size',A.size())
        
        B_nparray=(np.asarray(B_img)).astype(np.float32)
        B_nparray=scipy.ndimage.zoom(B_nparray,0.5,order=1)
        B_norm=(B_nparray-value1)/value2

        if self.transform is not None:
           B_norm = self.transform(image=B_norm)["image"]

        B_tensor=torch.from_numpy(B_norm).unsqueeze(0)
        B=B_tensor.repeat(3,1,1)

        # fig = plt.figure()
        # a = fig.add_subplot(2, 2, 1)
        # a.set_title('flip input image')
        # plt.imshow(A[0,:,:], cmap="gray")
        # plt.colorbar()

        # a = fig.add_subplot(2, 2, 2)
        # a.set_title('flip input image mask')
        # plt.imshow(maskA_tensor[0,:,:], cmap="gray")
        # plt.colorbar()

        # a = fig.add_subplot(2, 2, 3)
        # a.set_title('input non flip image')
        # plt.imshow(A_nparray, cmap="gray")
        # plt.colorbar()

        # a = fig.add_subplot(2, 2, 4)
        # a.set_title('flipped B image')
        # plt.imshow(B_norm, cmap="gray")
        # plt.colorbar()

        # plt.savefig('my_plot_A_B_maskA_debug.png')

        # print('B_img min max is ',B_img.getextrema())
        # print('B max after np is ', B_nparray.max())
        # print('B min after np is ', B_nparray.min())
        # print('B np shape', B_nparray.shape) 
        # print('B tensor max', torch.max(B))
        # print('B tensor min', torch.min(B))
        # print('tensor size',B.size())

        
        #fig = plt.figure()
        #a = fig.add_subplot(1, 1, 1)
        #a.set_title('Input image')
        #plt.imshow(A_norm, cmap="gray")
        #plt.colorbar()
        #plt.savefig('my_plotA_debug.png')
        

        #B = self.transform(B_img)
        # A=torch.from_numpy(np.array(A_img).astype(np.float32)).unsqueeze(0)
        
        #print('B shape', B.shape)
        #print('A max', A.max())
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        '''
        output the middle image
        '''
        # scipy.misc.imsave('crop.png', A.numpy()[0])
        return {'A': A, 'B': B, 'M_A': maskA_tensor,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
