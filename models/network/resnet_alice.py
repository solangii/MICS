import torch
import torch.nn as nn
from utils.mixup_utils import to_one_hot, get_lambda, middle_mixup_process
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, target=None, mix_type="vanilla", mixup_alpha=None, num_base_classes=-1,
                use_hard_positive_aug=False, add_noise_level=0., mult_noise_level=0., minimum_lambda=0.5,
                hpa_type="none", label_sharpening=True, label_mix="vanilla", label_mix_threshold=0.2,
                exp_coef=1., cutmix_prob=1., num_similar_class=3, classifiers=None,
                gaussian_h1=0.2, piecewise_linear_h1=0.5, piecewise_linear_h2=0., use_softlabel=True):

        if "mixup_hidden" in mix_type:
            layer_mix = random.randint(0, 3)
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
        out = F.relu(out)
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

        if layer_mix == 3:
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
        out = self.layer4(out)

        if target is not None:
            return out, target_reweighted, mix_label_mask
        else:
            return out


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet_alice(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
