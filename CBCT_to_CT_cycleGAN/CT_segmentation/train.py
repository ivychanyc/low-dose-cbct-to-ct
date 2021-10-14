# %%
import sys
import os
#from optparse import OptionParser
import numpy as np
import sys
import os
#from optparse import OptionParser
from argparse import ArgumentParser

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from utils import plot_img_and_mask

#%%
def train_net(net,
              epochs=1,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    #dir_img = 'data/train/'
    #dir_mask = 'data/train_masks/'
    #dir_checkpoint = 'checkpoints/'

    dir_img = '/data/Unpaired_MR_to_CT_Image_Synthesis-master/data/datasets/MR2CT/trainB/' ########please change here 
    dir_mask = '/data/Unpaired_MR_to_CT_Image_Synthesis-master/data/datasets/MR2CT/maskB/'
    dir_checkpoint = '/data/Unpaired_MR_to_CT_Image_Synthesis-master/checkpoints/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            #imgs = np.array([j[0] for j in b]).astype(np.float32) #10,256,256 #Ivy edited on 01.07.2021
            imgs = np.array([j[0] for j in b]).astype(np.float32) #10,256,256
            true_masks = np.array([j[1] for j in b])

            imgs = torch.from_numpy(imgs)
            #print('imgs torch shape',imgs.shape)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs) #Ivy edited on 30.06.2021
           # masks_pred = net(imgs[None, ...])
            masks_probs = F.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #plot_img_and_mask(imgs.cpu().numpy()[0,0,:,:], true_masks.cpu().numpy()[0,0,:,:])
            #if epoch==1:
            '''
            fig = plt.figure()
            a = fig.add_subplot(1, 5, 1)
            a.set_title('Input image')
            plt.imshow(imgs.cpu().numpy()[0,0,:,:])
            plt.colorbar()

            b = fig.add_subplot(1, 5, 2)
            b.set_title('target mask')
            plt.imshow(true_masks.cpu().numpy()[0,0,:,:])
            plt.colorbar()

            c = fig.add_subplot(1, 5, 3)
            c.set_title('Output mask')
            plt.imshow(masks_pred.detach().cpu().numpy()[0,0,:,:])
            plt.colorbar()

            d = fig.add_subplot(1, 5, 4)
            d.set_title('Output mask')
            plt.imshow(masks_probs.detach().cpu().numpy()[0,0,:,:])
            plt.colorbar()


            e = fig.add_subplot(1, 5, 5)
            e.set_title('Output mask treshold')
            plt.imshow(masks_probs.detach().cpu().numpy()[0,0,:,:]>0.5)
            plt.colorbar()
            plt.show()
            
            
            plt.savefig('my_plot2.png')
            '''
         
            
        

        print('Epoch finished ! Loss: {}'.format(epoch_loss / (i+1)))
        #
        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    '''
    #parser = OptionParser()
    
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    (options, args) = parser.parse_args()
    return options
    '''
    parser = ArgumentParser() 
    parser.add_argument('-e', '--epochs', dest='epochs', default=5, type=int,
                      help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batchsize', default=10,
                      type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', default=0.1,
                      type=float, help='learning rate')
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_argument('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_argument('-s', '--scale', dest='scale', type=float,
                      default=0.5, help='downscaling factor of the images')

    args, options = parser.parse_known_args()
    return args

#%%
if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

# %%
