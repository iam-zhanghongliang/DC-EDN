#coding:utf-8
#python validate.py --exp_id cu-net-0  --bs 1
from __future__ import division
from torch.autograd import Variable
import time
import json
import pdb
import shutil
import scipy.io
import sys, os, time
from PIL import Image, ImageDraw
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parameter import Parameter

from options.train_options import TrainOptions
from data.face_bbx import FACE
from data.picture_load import LoadPicture
from model.DC_EDN_model import dc_edn
from utils.util import AverageMeter
from utils.util import TrainHistoryFace, get_n_params, get_n_trainable_params, get_n_conv_params
from utils.visualizer import Visualizer
from utils.checkpoint import Checkpoint
from utils.logger import Logger
from utils.util import AdjustLR
from pylib import FaceAcc, Evaluation, HumanAug
cudnn.benchmark = True
from utils import auc_v1
def main():
    opt = TrainOptions().parse() 
    train_history = TrainHistoryFace()
    checkpoint = Checkpoint()
    visualizer = Visualizer(opt)
    exp_dir = os.path.join(opt.exp_dir, opt.exp_id)
    log_name = opt.vis_env + 'log.txt'
    visualizer.log_name = os.path.join(exp_dir, log_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_classes = 98
    layer_num = 4
    net = dc_edn(neck_size=4, growth_rate=32, init_chan_num=128,class_num=num_classes, layer_num=layer_num,order=1, loss_num=layer_num)
    net = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.lr, alpha=0.99,
                                    eps=1e-8, momentum=0, weight_decay=0)
    #image path of WFLW dataset 
    img_folder = ("/home/ylp/zhanghl/zhanghl_WLFW/WFLW_pre/WFLW_images")
    #load the test set .json file.
    img=LoadPicture("/home/ylp/zhanghl/DC-EDN/validate_dataset/list_98pt_test.txt.json" , img_folder, is_train=False)
    
    #load the model 
    checkpoints = torch.load("/home/ylp/zhanghl/DC-EDN/model/lr-0.0005-42-model-best.pth.tar")  
    #print(checkpoints.keys())
    net.load_state_dict(checkpoints['state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer'])
    net = net.cuda()
	# switch to evaluate mode
    net.eval()	
	#load the data
    val_loader = torch.utils.data.DataLoader(
        FACE( "/home/ylp/zhanghl/DC-EDN/validate_dataset/list_98pt_test.txt.json", "/home/ylp/zhanghl/zhanghl_WLFW/WFLW_pre/WFLW_images", is_train=False),
        batch_size=opt.bs, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True) #.json file of test set and images of test set.
    
    print("valation is staring ;; ;;;;;;;;;")
    #visualizer.log_path = os.path.join(opt.exp_dir, opt.exp_id, 'val_log.txt')
    val_rmse, predictions = validate(val_loader, net,visualizer, num_classes)    
    print("the val_rms is :{rms}".format(rms=val_rmse))
    
    '''
    #显示效果
    preds = predictions.numpy()
    #print(preds)      	
    for index,temp in enumerate(img):
        plt.figure("Image") # 图像窗口名称
        plt.imshow(temp)
        plt.axis('on') # 关掉坐标轴为 off
        plt.title('image') # 图像题目
        plt.scatter(preds[index,:, 0], preds[index,:, 1], s=10, marker='.', c='r')
        #plt.scatter(preds[index,60, 0], preds[index,60, 1], s=10, marker='.', c='r')
        #plt.scatter(preds[index,72, 0], preds[index,72, 1], s=10, marker='.', c='r')
        #plt.pause(0.001)
        plt.show()
        if index>20:
	        break
    '''       
      
def validate(val_loader, net, visualizer, num_classes):
    batch_time = AverageMeter()
    losses_det = AverageMeter()
    losses = AverageMeter()
    rmses0 = AverageMeter()
    rmses1 = AverageMeter()
    rmses2 = AverageMeter()
    inp_batch_list = []
    pts_batch_list = []
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    timeall=[]
    tmp = 0
    rmse_list = []
    for i, (inp, heatmap, pts, index, center, scale) in enumerate(val_loader):
        #print(inp.size())
        #print(pts.size())
        # input and groundtruth
        input_var = torch.autograd.Variable(inp, volatile=True)

        heatmap = heatmap.cuda(async=True)
        target_var = torch.autograd.Variable(heatmap)

        # output and loss
        #output1, output2 = net(input_var)
        #loss = (output1 - target_var) ** 2 + (output2 - target_var) ** 2
        try:
            time0 = time.time()
            output1 = net(input_var)
            time1 = time.time()
            timed = time1-time0
            tmp = tmp+1
            #print(timed*1000," ms")
            timeall.append(timed*1000)
        except:
            continue
        loss = 0
        for per_out in output1:
            tmp_loss = (per_out - target_var) ** 2
            loss = loss + tmp_loss.sum() / tmp_loss.numel()
        # calculate measure
        output = output1[-1].data.cpu()
        preds = Evaluation.final_preds(output, center, scale, [64, 64])
        rmse = np.sum(FaceAcc.per_image_rmse(preds.numpy(), pts.numpy())) / inp.size(0)
        rmse_list.append(rmse)
        rmses2.update(rmse, inp.size(0))
        for n in range(output.size(0)):
            predictions[index[n], :, :] = preds[n, :, :]
    auc_v1.AUCError(rmse_list, 0.10, step=0.0001, showCurve=True)	
    return rmses2.avg, predictions



if __name__ == '__main__':
    main()
