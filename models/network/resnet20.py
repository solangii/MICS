import torch
import torch.nn as nn
from utils.mixup_utils import to_one_hot, middle_mixup_process, get_lambda
from torch.autograd import Variable
import numpy as np
import random


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)
        # self.avgpool = nn.AvgPool2d(8, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, target=None, mix_type="vanilla", mixup_alpha=None, num_base_classes=-1,
                use_hard_positive_aug=False, add_noise_level=0., mult_noise_level=0., minimum_lambda=0.5,
                hpa_type="none", label_sharpening=True, label_mix="vanilla", label_mix_threshold=0.2,
                exp_coef=1., cutmix_prob=1., num_similar_class=3, classifiers=None,
                gaussian_h1=0.2, piecewise_linear_h1=0.5, piecewise_linear_h2=0., use_softlabel=True):

        if "mixup_hidden" in mix_type:
            layer_mix = random.randint(0, 2)
        else:
            layer_mix = None

        out = x

        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)

            # https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py#L243
            if use_hard_positive_aug:
                lam = max(lam, 1 - lam)
                lam = max(lam, minimum_lambda)

            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = Variable(lam)

        if target is not None:
            target_reweighted = to_one_hot(target, self.num_classes)

        if layer_mix == 0:
            out, target_reweighted, mix_label_mask = middle_mixup_process(out, target_reweighted, num_base_classes,
                                                                          lam,
                                                                          use_hard_positive_aug, add_noise_level,
                                                                          mult_noise_level,
                                                                          hpa_type, label_sharpening, label_mix,
                                                                          label_mix_threshold,
                                                                          exp_coef=exp_coef,
                                                                          gaussian_h1=gaussian_h1,
                                                                          piecewise_linear_h1=piecewise_linear_h1,
                                                                          piecewise_linear_h2=piecewise_linear_h2,
                                                                          use_softlabel=use_softlabel)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)

        if layer_mix == 1:
            out, target_reweighted, mix_label_mask = middle_mixup_process(out, target_reweighted, num_base_classes,
                                                                          lam,
                                                                          use_hard_positive_aug, add_noise_level,
                                                                          mult_noise_level,
                                                                          hpa_type, label_sharpening, label_mix,
                                                                          label_mix_threshold,
                                                                          exp_coef=exp_coef,
                                                                          gaussian_h1=gaussian_h1,
                                                                          piecewise_linear_h1=piecewise_linear_h1,
                                                                          piecewise_linear_h2=piecewise_linear_h2,
                                                                          use_softlabel=use_softlabel)

        out = self.layer2(out)

        if layer_mix == 2:
            out, target_reweighted, mix_label_mask = middle_mixup_process(out, target_reweighted, num_base_classes,
                                                                          lam,
                                                                          use_hard_positive_aug, add_noise_level,
                                                                          mult_noise_level,
                                                                          hpa_type, label_sharpening, label_mix,
                                                                          label_mix_threshold,
                                                                          exp_coef=exp_coef,
                                                                          gaussian_h1=gaussian_h1,
                                                                          piecewise_linear_h1=piecewise_linear_h1,
                                                                          piecewise_linear_h2=piecewise_linear_h2,
                                                                          use_softlabel=use_softlabel)

        out = self.layer3(out)

        if target is not None:
            return out, target_reweighted, mix_label_mask
        else:
            return out


def resnet20(**kwargs):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    return model
