from models.network.resnet18 import *
from models.network.resnet20 import *
from models.network.resnet_alice import *
from utils.mixup_utils import to_one_hot

class MYNET(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        if self.args.dataset == 'cifar100':
            self.encoder = resnet20(num_classes=self.args.num_classes)
            self.num_features = 64
        if self.args.dataset == 'mini_imagenet':
            self.encoder = resnet18(False, num_classes=self.args.num_classes)
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, num_classes=self.args.num_classes)
            self.num_features = 512
        if self.args.use_resnet_alice:
            self.encoder = resnet_alice(False, num_classes=self.args.num_classes)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
        nn.init.orthogonal_(self.fc.weight)

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward_metric(self, x):
        x = self.encode(x)
        x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
        x = self.args.temperature * x
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def forward_mix(self, args, x, target=None, hpa_type="none"):
        cur_num_class = int(max(target)) + 1
        if args.use_mixup:
            x, retarget, mix_label_mask = self.encoder(x, target, args.train, args.mixup_alpha,
                                                       cur_num_class, args.use_hard_positive_aug, args.add_noise_level,
                                                       args.mult_noise_level, args.minimum_lambda, hpa_type,
                                                       args.label_sharpening, args.label_mix, args.label_mix_threshold,
                                                       exp_coef=args.exp_coef,
                                                       num_similar_class=args.num_similar_class,
                                                       classifiers=self.fc.weight,
                                                       gaussian_h1=args.gaussian_h1,
                                                       piecewise_linear_h1=args.piecewise_linear_h1,
                                                       piecewise_linear_h2=args.piecewise_linear_h2,
                                                       use_softlabel=args.use_softlabel)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

            if args.use_midpoint:
                classifier = self.calculate_middle_classifier(mix_label_mask, args.normalized_middle_classifier)
                x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(classifier, p=2, dim=-1))
                x = self.args.temperature * x  # 25, 72
        else:
            x = self.encode(x)
            retarget = to_one_hot(target, self.pre_allocate)
        return x, retarget

    def get_logits(self, x, fc):
        return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def calculate_middle_classifier(self, mix_label_mask, normalized_middle_classifier=True):
        if normalized_middle_classifier:
            first_classifier = F.normalize(self.fc.weight[mix_label_mask[0]], p=2, dim=-1)
            second_classifier = F.normalize(self.fc.weight[mix_label_mask[1]], p=2, dim=-1)
        else:
            first_classifier = self.fc.weight[mix_label_mask[0]]
            second_classifier = self.fc.weight[mix_label_mask[1]]

        middle_classifier = (first_classifier + second_classifier) / 2
        return middle_classifier
