# -*- coding:utf-8 -*-
# Zhiqiang Tang, May 2017
import os, sys
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from utils import imutils
from pylib import HumanPts, FaceAug, HumanAug, FacePts

def sample_from_bounded_gaussian(x):
    return max(-2*x, min(2*x, np.random.randn()*x))

class LoadPicture(data.Dataset):
    def __init__(self, jsonfile, img_folder, inp_res=256, out_res=64, is_train=True, sigma=1,
                 scale_factor=0.25, rot_factor=30, std_size=200):  #inp_res=256 原始,out_res=64

        self.img_folder = img_folder  # root image folders
        self.is_train = is_train  # training set or test set
        self.inp_res = inp_res  #图像大小
        self.out_res = out_res  #64 对应heatmap大小
        self.sigma = sigma  #高斯热图对应方差
        self.scale_factor = scale_factor  #缩放参数
        self.rot_factor = rot_factor  #旋转参数
        self.std_size = std_size   #旋转参数

        # create train/val split
        with open(jsonfile, 'r') as anno_file:
            self.anno = json.load(anno_file)
            print ('loading json file is done...')
        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            #if val['dataset'] != '300w_cropped':
            if val['isValidation'] == True :#or val['dataset'] == 'ibug':
                self.valid.append(idx)
            else:
                self.train.append(idx)

        # self.mean, self.std = self._compute_mean()
        if self.is_train:
            print ('total training images: ', len(self.train))
        else:
            print ('total validation images: ', len(self.valid))

    def _compute_mean(self):
        meanstd_file = 'dataset/face.pth.tar'
        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train:
                a = self.anno[index]
                img_path = os.path.join(self.img_folder, a['img_paths'])
                img = imutils.load_image(img_path)  # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
            }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def color_normalize(self, x, mean, std):
        if x.size(0) == 1:
            x = x.repeat(3, x.size(1), x.size(2))

        for t, m, s in zip(x, mean, std):
            t.sub_(m).div_(s)
        return x

    def __getitem__(self, index):
        #print("the index is '''''' ")
        #print(index)

        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]
        if a['pts_paths'] == 'ibug/image_092 _01.pts':
            a['pts_paths'] = 'ibug/image_092_01.pts'
            a['img_paths'] = 'ibug/image_092_01.jpg'
        img_path = os.path.join(self.img_folder, a['image_paths'])
        pts_path = os.path.join(self.img_folder, a['pts_paths'])
        if pts_path[-4:] == '.txt':
            pts = np.loadtxt(pts_path)  # L x 2
        elif pts_path[-4:] == '.pts':
            pts = FacePts.Pts2Lmk(pts_path)  # L x 2

        pts = torch.Tensor(pts)
        #assert torch.sum(pts - torch.Tensor(a['pts'])) == 0
        s = torch.Tensor([a['scale_provided_det']]) * 1.1
        c = torch.Tensor(a['objpos_det'])
        img = scipy.misc.imread(img_path, mode='RGB') # CxHxW
        #img = torch.from_numpy(img)
        return img

        