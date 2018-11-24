from collections import namedtuple

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import math


# class Vgg16(nn.Module):
#
#     def __init__(self , reg, requires_grad):
#         super(Vgg16, self).__init__()
#         vgg = models.vgg16(pretrained=True)
#         vgg_pretrained_features=vgg.features
#         self.reg=reg
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#
#         self.regressor=nn.Linear(8192,1)
#
#         self._initialize_weights()
#
#         for x in range(4):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(4, 9):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(9, 16):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(16, 23):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
#
#     def forward(self, X):
#         h = self.slice1(X)
#         h_relu1_2 = h
#         h = self.slice2(h)
#         h_relu2_2 = h
#         h = self.slice3(h)
#         h_relu3_3 = h
#         h = self.slice4(h)
#         h_relu4_3 = h
#         vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
#         feats = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
#         if self.reg:
#             x = F.adaptive_avg_pool2d(h_relu4_3, (4, 4))
#             x = x.view(x.size(0), -1)
#             x = F.dropout(x, training=self.training)
#             x = self.regressor(x)
#             return x,feats
#
#         return feats

class Vgg16(nn.Module):

    def __init__(self , reg, requires_grad):
        super(Vgg16, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        vgg_pretrained_features=vgg.features
        self.reg=reg
        self.slice0 = torch.nn.Sequential()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        self.deconv1 = UpsampleConvLayer(256, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1,
                                         upsample_size=(320))  # upsample_size=(450,450)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()


        self.regressor=nn.Linear(512,1)

        # self._initialize_weights()

        for x in range(3):
            self.slice0.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 6):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 13):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(13, 23):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 33):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(33, 43):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, X):
        h = self.slice0(X)
        h_relu1_1 = h
        h = self.slice1(h)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        # h = self.slice5(h)
        #         # h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        feats = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)

        score=None
        # stream 1
        if self.reg:
            x = F.adaptive_avg_pool2d(h_relu4_3, (1, 1))
            x = x.view(x.size(0), -1)
            x = F.dropout(x, training=self.training)
            score = self.regressor(x)

        # stream 2
        y=feats.relu3_3
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)

        return score,y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()





class Vgg16_fix(nn.Module):

    def __init__(self, requires_grad):
        super(Vgg16_fix, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        vgg_pretrained_features = vgg.features
        self.slice0 = torch.nn.Sequential()
        self.slice1 = torch.nn.Sequential()
        # self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()

        for x in range(3):
            self.slice0.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 6):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(6, 13):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(13, 23):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice0(X)
        h_relu1_1 = h
        h = self.slice1(h)
        h_relu1_2 = h
        # h = self.slice2(h)
        # h_relu2_2 = h
        # h = self.slice3(h)
        # h_relu3_3 = h
        # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3'])
        # feats = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3)
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2'])
        feats = vgg_outputs(h_relu1_2)
        return feats

    # def forward(self, X):
    #     h = self.slice1(X)
    #     if self.layer==1:
    #         s=self.regressing()
    #         return h
    #
    #     h = self.slice2(h)
    #     if self.layer==2:
    #         return h
    #
    #     h = self.slice3(h)
    #     if self.layer==3:
    #         return h
    #
    #     h = self.slice4(h)
    #     if self.layer==4:
    #         return h
    #
    #
    #
    # def regressing(self,x):
    #     x = F.adaptive_avg_pool2d(x, (4, 4))
    #     x = x.view(x.size(0), -1)
    #     x = F.dropout(x, training=self.training)
    #     x = self.regressor(x)


    def _initialize_weights(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class deconv_net(nn.Module):

    def __init__(self):
        super(deconv_net, self).__init__()
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample_size=(350))  #upsample_size=(450,450)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self,y):
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y




class distortion_net(nn.Module):

    def __init__(self):
        super(distortion_net, self).__init__()
        self.features=make_layers(cfg['MY'], batch_norm=False)
        self.regressor=nn.Linear(4096,1)
        self._initialize_weights()

    def forward(self, x, attention_map=None):
        feats = self.features(x)
        x = F.adaptive_avg_pool2d(feats,(4,4))
        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = self.regressor(x)
        return x,feats

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'MY':[64, 64, 'M', 128, 128, 'M', 256, 256, 256],
    'MY2': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None,upsample_size=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.upsample_size = upsample_size
        reflection_padding = kernel_size // 2
        self.interplate=nn.Upsample(scale_factor=upsample, size=upsample_size, mode='bilinear')
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.interplate(x_in)
        elif self.upsample_size:
            x_in = self.interplate(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


if __name__ == "__main__":
    vgg = Vgg16(requires_grad=False).cuda()