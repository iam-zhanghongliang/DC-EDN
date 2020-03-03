# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import torch.utils.checkpoint as cp
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter

class CSDN_Tem1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem1, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch,
            bias=False
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.depth_conv(out)
        out = self.point_conv(out)
        return out

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch,
            bias=False
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
    def forward(self, input):
        out2 = self.depth_conv(input)
        out1 = self.depth_conv(input)
        out1 = self.depth_conv(out1)
        out = out1 + out2
        out = self.point_conv(out)
        return out

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class _Adapter(nn.Module):
    def __init__(self, num_input_features, num_output_features, efficient):
        super(_Adapter, self).__init__()
        self.add_module('adapter_norm', nn.BatchNorm2d(num_input_features))
        self.add_module('adapter_relu', nn.ReLU(inplace=True))
        self.add_module('adapter_conv', CSDN_Tem1(num_input_features, num_output_features))
        #self.add_module('adapter_conv', nn.Conv2d(num_input_features, num_output_features,
                                                  #kernel_size=1, stride=1, bias=False))
        self.efficient = efficient
    def forward(self, prev_features):
        bn_function = _bn_function_factory(self.adapter_norm, self.adapter_relu,
                                           self.adapter_conv)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            adapter_output = cp.checkpoint(bn_function, *prev_features)
        else:
            adapter_output = bn_function(*prev_features)

        return adapter_output


class _DenseLayer(nn.Module):   #输入64*64 *128   dense_layer:norm->relu->bottleneck->conv->dropout
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', CSDN_Tem(bn_size * growth_rate, growth_rate)),
        self.drop_rate = drop_rate#Dropout,可选的,用于防止过拟合, out:64*64 *128
        self.efficient = efficient
    #bottleneck to Res2Net 以更细粒度（granular level）表示多尺度特征，并增加每个网络层的感受野
    def forward(self, prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_fea.requires_grad for prev_fea in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.Module):   #输入64*64 *128
    def __init__(self, in_num, growth_rate, neck_size, layer_num, max_link,
                 drop_rate=0, efficient=True, requires_skip=True, is_up=False):

        self.saved_features = []
        self.max_link = max_link
        self.requires_skip = requires_skip
        super(_DenseBlock, self).__init__()
        max_in_num = in_num + max_link * growth_rate
        self.final_num_features = max_in_num
        self.layers = nn.ModuleList()
        print('layer number is %d' % layer_num)
        for i in range(0, layer_num):
            if i < max_link:
                tmp_in_num = in_num + i * growth_rate  #前后两层间通道数只增加32，越往后层继续拼接，使用参数相对resnet较少
            else:
                tmp_in_num = max_in_num
            print('layer %d input channel number is %d' % (i, tmp_in_num))
            self.layers.append(_DenseLayer(tmp_in_num, growth_rate=growth_rate,
                                           bn_size=neck_size, drop_rate=drop_rate,
                                           efficient=efficient))
        # self.layers = nn.ModuleList(self.layers)

        self.adapters_ahead = nn.ModuleList()
        adapter_in_nums = []
        adapter_out_num = in_num
        if is_up:
            adapter_out_num = adapter_out_num // 2
        for i in range(0, layer_num):
            if i < max_link:
                tmp_in_num = in_num + (i+1) * growth_rate
            else:
                tmp_in_num = max_in_num + growth_rate
            adapter_in_nums.append(tmp_in_num)
            print('adapter %d input channel number is %d' % (i, adapter_in_nums[i]))
            self.adapters_ahead.append(_Adapter(adapter_in_nums[i], adapter_out_num,
                                                efficient=efficient))
        # self.adapters_ahead = nn.ModuleList(self.adapters_ahead)
        print('adapter output channel number is %d' % adapter_out_num)

        if requires_skip:
            print('creating skip layers ...')
            self.adapters_skip = nn.ModuleList()
            for i in range(0, layer_num):
                self.adapters_skip.append(_Adapter(adapter_in_nums[i], adapter_out_num,
                                                   efficient=efficient))
            # self.adapters_skip = nn.ModuleList(self.adapters_skip)

    def forward(self, x, i):
        if i == 0:
            self.saved_features = []

        if type(x) is torch.Tensor:
            x = [x]
        if type(x) is not list:
            raise Exception('type(x) should be list, but it is: ', type(x))
        #err  Exception: ('type(x) should be list, but it is: ', <class 'torch.autograd.variable.Variable'>)
        # for t_x in x:
            # print 't_x type: ', type(t_x)
            # print 't_x size: ', t_x.size()

        x = x + self.saved_features
        # for t_x in x:
        #     print 't_x type: ', type(t_x)
        #     print 't_x size: ', t_x.size()

        out = self.layers[i](x)
        if i < self.max_link:
            self.saved_features.append(out)
        elif len(self.saved_features) != 0:
            self.saved_features.pop(0)
            self.saved_features.append(out)
        x.append(out)
        out_ahead = self.adapters_ahead[i](x)
        if self.requires_skip:
            out_skip = self.adapters_skip[i](x)
            return out_ahead, out_skip
        else:
            return out_ahead

class _IntermediaBlock(nn.Module):  #64*64 *128  中间监督 类似于过渡层，能提高一点精度，why？
    def __init__(self, in_num, out_num, layer_num, max_link, efficient=True):  #max_link是特征层向后连接的跳跃数

        max_in_num = in_num + max_link * out_num  #相同大小特征图通道拼接，原dense是前面所有层的通道拼接
        self.final_num_features = max_in_num
        self.saved_features = []
        self.max_link = max_link
        super(_IntermediaBlock, self).__init__()
        print('creating intermedia block ...')
        self.adapters = nn.ModuleList()
        for i in range(0, layer_num-1):
            if i < max_link:
                tmp_in_num = in_num + (i+1) * out_num
            else:
                tmp_in_num = max_in_num
            print('intermedia layer %d input channel number is %d' % (i, tmp_in_num))
            self.adapters.append(_Adapter(tmp_in_num, out_num, efficient=efficient))
        # self.adapters = nn.ModuleList(self.adapters)
        print('intermedia layer output channel number is %d' % out_num)

    def forward(self, x, i):
        if i == 0:
            self.saved_features = []
            if type(x) is torch.Tensor:
                if self.max_link != 0:
                    self.saved_features.append(x)
            elif type(x) is list:
                if self.max_link != 0:
                    self.saved_features = self.saved_features + x
            return x

        if type(x) is torch.Tensor:
            x = [x]
        if type(x) is not list:
            raise Exception('type(x) should be list, but it is: ', type(x))

        x = x + self.saved_features
        #print("#########################################################")
        #print("i-1val:",i-1)
        out = self.adapters[i-1](x)
        if i < self.max_link:
            self.saved_features.append(out)
        elif len(self.saved_features) != 0:
            self.saved_features.pop(0)
            self.saved_features.append(out)
        # print('middle list length is %d' % len(self.saved_features))
        return out

class _Bn_Relu_Conv1x1(nn.Sequential):  #64*64 *128
    def __init__(self, in_num, out_num):
        super(_Bn_Relu_Conv1x1, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_num, out_num, kernel_size=1,
                                          stride=1, bias=False))



class _CU_Net(nn.Module):   #64*64 *128  作为hg输入
    def __init__(self, in_num, neck_size, growth_rate, layer_num, max_link):
        super(_CU_Net, self).__init__()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.num_blocks = 4  #4个hg的堆叠，最多16个，中间有过渡层(conv,pool),之前一直改训练的layernum值是一个Block中有n个 layer
        print('creating hg ...')
        for i in range(0, self.num_blocks):
            print('creating down block %d ...' % i) #_DenseBlock包含botteneck ,down_up,down: requires_skip=True
            self.down_blocks.append(_DenseBlock(in_num=in_num, neck_size=neck_size,
                                    growth_rate=growth_rate, layer_num=layer_num,
                                    max_link=max_link, requires_skip=True))  #_DenseBlock中的_DenseLayer:botteneck可以改变
            print('creating up block %d ...' % i)
            self.up_blocks.append(_DenseBlock(in_num=in_num*2, neck_size=neck_size,growth_rate=growth_rate, layer_num=layer_num, 
									max_link=max_link, requires_skip=False, is_up=True))
        # self.down_blocks = nn.ModuleList(self.down_blocks)
        # self.up_blocks = nn.ModuleList(self.up_blocks)
        print('creating neck block ...')
        self.neck_block = _DenseBlock(in_num=in_num, neck_size=neck_size,
                                      growth_rate=growth_rate, layer_num=layer_num,
                                      max_link=max_link, requires_skip=False) #64*64 *128
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) #32*32 *128
        self.upsample = nn.Upsample(scale_factor=2)  #上采样扩大2倍 64*64 *128

    def forward(self, x, i):
        skip_list = [None] * self.num_blocks
        # print 'input x size is ', x.size()
        for j in range(0, self.num_blocks):
            # print('using down block %d ...' % j)
            x, skip_list[j] = self.down_blocks[j](x, i)
            # print 'output size is ', x.size()
            # print 'skip size is ', skip_list[j].size()
            x = self.maxpool(x)
        # print('using neck block ...')
        x = self.neck_block(x, i)
        # print 'output size is ', x.size()
        for j in list(reversed(range(0, self.num_blocks))):
            x = self.upsample(x)
            # print('using up block %d ...' % j)
            x = self.up_blocks[j]([x, skip_list[j]], i)
            # print 'output size is ', x.size()
        return x

class dc_edn_model(nn.Module):
    def __init__(self, init_chan_num, neck_size, growth_rate,
                 class_num, layer_num, order, loss_num):   #128，4，32，68，2，1,2
        assert loss_num <= layer_num and loss_num >= 1
        loss_every = float(layer_num) / float(loss_num)
        self.loss_anchors = []
        for i in range(0, loss_num):
            tmp_anchor = int(round(loss_every * (i+1)))
            if tmp_anchor <= layer_num:
                self.loss_anchors.append(tmp_anchor)

        assert layer_num in self.loss_anchors
        assert loss_num == len(self.loss_anchors)

        if order >= layer_num:
            print ('order is larger than the layer number.')  #order大于layer number 不符合则退出
            exit()
        print('layer number is %d' % layer_num)
        print('loss number is %d' % loss_num)
        print('loss anchors are: ', self.loss_anchors)
        print('order is %d' % order)
        print('growth rate is %d' % growth_rate)
        print('neck size is %d' % neck_size)
        print('class number is %d' % class_num)
        print('initial channel number is %d' % init_chan_num)
        num_chans = init_chan_num
        super(dc_edn_model, self).__init__() #face input 256*256,Feature block,in_channel=3, init_chan_num=128，
        self.layer_num = layer_num    #2
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, init_chan_num, kernel_size=7, stride=2, padding=3, bias=False)),#out:128*128 *128
            ('norm0', nn.BatchNorm2d(init_chan_num)),  #out:128*128 *128
            ('relu0', nn.ReLU(inplace=True)),  #out:128*128 *128
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),  #这无padding, out:64*64 *128  一般kernel_size=3, stride=2, padding=1
        ]))
        # self.denseblock0 = _DenseBlock(layer_num=4, in_num=init_chan_num,
        #                                neck_size=neck_size, growth_rate=growth_rate)
        # hg_in_num = init_chan_num + growth_rate * 4
        print('channel number is %d' % num_chans)
        self.hg = _CU_Net(in_num=num_chans, neck_size=neck_size, growth_rate=growth_rate,
                             layer_num=layer_num, max_link=order)#64*64 *128
        #_CU_Net与dense_unet: _Hourglass hg 的构造一样，只是命名不同，一般堆叠的unet之间有linears连接
        self.linears = nn.ModuleList()
        for i in range(0, layer_num):  #这是dense_unet: _Hourglass没有的，在layer_num内考虑linears，why？
            self.linears.append(_Bn_Relu_Conv1x1(in_num=num_chans, out_num=class_num))  #64*64 *128
        # self.linears = nn.ModuleList(self.linears)
        # intermedia_in_nums = []
        # for i in range(0, num_units-1):
        #     intermedia_in_nums.append(num_chans * (i+2))
        self.intermedia = _IntermediaBlock(in_num=num_chans, out_num=num_chans,
                                           layer_num=layer_num, max_link=order)#这是dense_unet: _Hourglass没有的中间监督

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d): # conv一般卷积，核大小<特征图大小，nn.ConvTranspose2d上采样卷积,对特征图补0，核大小>特征图大小
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels  #n :k*k*in_channels 为当前卷积层参数的个数
                stdv = 1/math.sqrt(n)    #对卷积层参数量n进行开平方运算 ,来作为卷积核大小K*K权重归一化的范围区间
                m.weight.data.uniform_(-stdv, stdv) #系统默认权重初始化方法，这里是用conv,一般有Linear，Conv，BN，默认偏转不存在时，
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
                    # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                # m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        # print(x.size())
        x = self.features(x)
        # print(x.size())
        # x = self.denseblock0(x)
        # print 'x size is', x.size()
        out = []
        # middle = []
        # middle.append(x)
        for i in range(0, self.layer_num):
            # print('using intermedia layer %d ...' % i)
            x = self.intermedia(x, i)
            # print 'x size after intermedia layer is ', x.size()
            # print('using hg %d ...' % i)
            x = self.hg(x, i)
            # print 'x size after hg is ', x.size()
            # middle.append(x)
            if (i+1) in self.loss_anchors:
                tmp_out = self.linears[i](x)
                # print 'tmp output size is ', tmp_out.size()
                out.append(tmp_out)
            # if i < self.num_units-1:
        # exit()
        assert len(self.loss_anchors) == len(out)
        return out

def dc_edn(neck_size, growth_rate, init_chan_num,
                  class_num, layer_num, order, loss_num):

    net = dc_edn_model(init_chan_num=init_chan_num, neck_size=neck_size,
                          growth_rate=growth_rate, class_num=class_num,
                          layer_num=layer_num, order=order, loss_num=loss_num)
    return net

