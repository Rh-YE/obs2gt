"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from ..util import get_ckpt_path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True, use_random_weights=False, input_channels=3):
        super().__init__()
        self.input_channels = input_channels
        self.scaling_layer = ScalingLayer(input_channels)
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=not use_random_weights, requires_grad=False, input_channels=input_channels)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        
        # 当使用随机权重时，初始化线性层
        if use_random_weights:
            for lin in [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]:
                lin.model.apply(lambda x: torch.nn.init.normal_(x.weight) if hasattr(x, 'weight') else None)
        # 只有在不使用随机权重时才加载预训练模型
        else:
            self.load_from_pretrained()
        
        # 确保所有参数都被冻结，包括新创建的参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 再次确认VGG网络的参数被冻结
        for param in self.net.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "sgm/modules/autoencoding/lpips/loss")
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips", use_random_weights=False, input_channels=3):
        if name != "vgg_lpips" and not use_random_weights:
            raise NotImplementedError
        model = cls(use_random_weights=use_random_weights, input_channels=input_channels)
        if not use_random_weights and input_channels == 3:
            ckpt = get_ckpt_path(name)
            model.load_state_dict(
                torch.load(ckpt, map_location=torch.device("cpu")), strict=False
            )
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
            
        # 确保返回的是标量值，对批次维度求平均
        val = val.mean()
        return val


class ScalingLayer(nn.Module):
    def __init__(self, input_channels=3):
        super(ScalingLayer, self).__init__()
        self.input_channels = input_channels
        
        if input_channels == 3:
            # 原始RGB的归一化参数
            shift = [-0.030, -0.088, -0.188]
            scale = [0.458, 0.448, 0.450]
        else:
            # 对于非RGB输入，使用零均值和单位方差
            shift = [0.0] * input_channels
            scale = [1.0] * input_channels
            
        self.register_buffer(
            "shift", torch.Tensor(shift)[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor(scale)[None, :, None, None]
        )

    def forward(self, inp):
        # return (inp - self.shift) / self.scale
        return inp


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, input_channels=3):
        super(vgg16, self).__init__()
        self.input_channels = input_channels
        
        if input_channels == 3:
            # 使用标准的预训练VGG16
            vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        else:
            # 创建修改过的VGG16，第一层适应不同的输入通道数
            vgg_pretrained_features = models.vgg16(pretrained=False).features
            if pretrained and input_channels != 3:
                # 如果需要预训练权重但输入通道不是3，我们需要调整第一层
                original_vgg = models.vgg16(pretrained=True)
                original_first_conv = original_vgg.features[0]
                
                # 创建新的第一层卷积
                new_first_conv = nn.Conv2d(
                    input_channels, 64, kernel_size=3, stride=1, padding=1
                )
                
                # 初始化新的第一层权重
                if input_channels < 3:
                    # 如果输入通道少于3，取RGB权重的前几个通道
                    new_first_conv.weight.data = original_first_conv.weight.data[:, :input_channels, :, :]
                elif input_channels > 3:
                    # 如果输入通道多于3，重复RGB权重或随机初始化额外通道
                    new_first_conv.weight.data[:, :3, :, :] = original_first_conv.weight.data
                    # 额外通道使用随机初始化或复制现有通道
                    for i in range(3, input_channels):
                        new_first_conv.weight.data[:, i:i+1, :, :] = original_first_conv.weight.data[:, i%3:i%3+1, :, :]
                
                new_first_conv.bias.data = original_first_conv.bias.data
                vgg_pretrained_features[0] = new_first_conv
                
                # 加载其余预训练权重
                for i in range(1, len(vgg_pretrained_features)):
                    if hasattr(vgg_pretrained_features[i], 'weight'):
                        vgg_pretrained_features[i].weight.data = original_vgg.features[i].weight.data
                    if hasattr(vgg_pretrained_features[i], 'bias') and vgg_pretrained_features[i].bias is not None:
                        vgg_pretrained_features[i].bias.data = original_vgg.features[i].bias.data
            elif not pretrained:
                # 如果不使用预训练权重，直接修改第一层
                vgg_pretrained_features[0] = nn.Conv2d(
                    input_channels, 64, kernel_size=3, stride=1, padding=1
                )
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # 确保所有参数都被正确设置requires_grad
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)
