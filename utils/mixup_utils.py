from torch.autograd import Variable
import matplotlib
from utils.labels import *

matplotlib.use('agg')


def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

    return Variable(y_onehot.cuda(), requires_grad=False)


def intra_class_permuted_indices(one_hot_label):
    label = torch.argmax(one_hot_label, dim=1)
    indices = [int(np.random.permutation((label == label[i]).nonzero().cpu())[0])
               for i in range(one_hot_label.size(0))]
    return indices


def label_mix_process(label1, label2, num_base_classes, lam, label_mix_type="vanilla",
                      label_mix_threshold=0.2, exp_coef=1., gaussian_h1=0.2,
                      piecewise_linear_h1=0.5, piecewise_linear_h2=0.):
    label = label1 * lam + label2 * (1 - lam)

    # Steep mix
    if "steep" in label_mix_type:
        intra_mix_indices = (label == 1.).sum(dim=-1).nonzero().squeeze()
        slope = 1 / (1 - label_mix_threshold)

        # Calculate labels
        y1 = (lam - label_mix_threshold) * slope if label_mix_threshold < lam else 0
        y2 = (1 - lam - label_mix_threshold) * slope if label_mix_threshold < 1 - lam else 0

        if label_mix_type == "steep_other":
            num_divide = num_base_classes - 2
        else:
            num_divide = label1.shape[-1] - num_base_classes
        y3 = (1 - y1 - y2) / num_divide

        # Create maskers
        mask = (label == 0.)
        mask[intra_mix_indices, :] = False
        if label_mix_type == "steep_other":
            mask[:, num_base_classes:] = False
        else:
            mask[:, :num_base_classes] = False

        # Combine labels
        label3 = torch.ones(label1.size()).cuda()
        label3[~mask] = 0.

        label = label1 * y1 + label2 * y2 + label3 * y3
        label[intra_mix_indices] = label1[intra_mix_indices]

    elif "dummy" in label_mix_type:
        intra_mix_indices = (label == 1.).sum(dim=-1).nonzero().squeeze()

        # Exponential label mix
        if "exp_dummy" == label_mix_type:
            y1 = exponential_func(lam, exp_coef)
            y2 = exponential_func(1 - lam, exp_coef)

        # Gaussian label mix
        elif "gaussian_dummy" == label_mix_type:
            y1 = adjusted_gaussian_func(lam.cpu(), gaussian_h1)
            y2 = adjusted_gaussian_func((1 - lam).cpu(), gaussian_h1)

        # Sine label mix
        elif "sine_dummy" == label_mix_type:
            y1 = sine_func(lam.cpu())
            y2 = sine_func((1 - lam).cpu())

        # Piecewise linear label mix
        elif "piecewise_linear_dummy" == label_mix_type:
            y1 = piecewise_func(lam.cpu(), piecewise_linear_h1, piecewise_linear_h2)
            y2 = piecewise_func((1 - lam).cpu(), piecewise_linear_h1, piecewise_linear_h2)

        else:
            raise AssertionError(f"There is no mix type: {label_mix_type}")

        num_divide = label1.shape[-1] - num_base_classes
        y3 = (1 - y1 - y2) / num_divide

        # Create maskers
        mask = (label == 0.)
        mask[intra_mix_indices, :] = False
        mask[:, :num_base_classes] = False

        # Combine labels
        label3 = torch.ones(label1.size()).cuda()
        label3[~mask] = 0.

        label = label1 * y1 + label2 * y2 + label3 * y3
        label[intra_mix_indices] = label1[intra_mix_indices]

    return label


def get_middle_label(label1, label2, num_base_classes):
    label1 = np.argmax(label1.cpu().numpy(), axis=1)
    label2 = np.argmax(label2.cpu().numpy(), axis=1)

    mix_label_hash = dict()
    middle_label = list()
    counter = 0

    for i in range(num_base_classes):
        mix_label_hash[(i, i)] = counter
        counter += 1

    for i in range(label1.shape[0]):
        target_tuple = (min(label1[i], label2[i]), max(label1[i], label2[i]))

        if target_tuple not in mix_label_hash:
            mix_label_hash[target_tuple] = counter
            counter += 1

        middle_label.append(mix_label_hash[target_tuple])

    mix_label_mask = list()
    for i in range(2):
        mix_label_mask.append([int(key[i]) for key in mix_label_hash.keys()])

    return mix_label_mask, to_one_hot(torch.tensor(middle_label), len(mix_label_hash))


def middle_label_mix_process(label1, label2, num_base_classes, lam, label_mix_type,
                             label_mix_threshold=0.2, exp_coef=1., gaussian_h1=0.2,
                             piecewise_linear_h1=0.5, piecewise_linear_h2=0., use_softlabel=True):
    assert label_mix_type != "vanilla", "Please use different option for label mix."
    assert "dummy" in label_mix_type, "Please use different option for label mix."

    mix_label_mask, label3 = get_middle_label(label1, label2, num_base_classes)
    if label3.size(1) > label1.size(1):
        zero_stack = torch.zeros([label1.size(0), label3.size(1) - label1.size(1)]).cuda()
    else:
        zero_stack = None

    if "steep_dummy" == label_mix_type:
        slope = 1 / (1 - label_mix_threshold)
        y1 = (lam - label_mix_threshold) * slope if label_mix_threshold < lam else 0
        y2 = (1 - lam - label_mix_threshold) * slope if label_mix_threshold < 1 - lam else 0

    # Exponential label mix
    elif "exp_dummy" == label_mix_type:
        y1 = exponential_func(lam, exp_coef)
        y2 = exponential_func(1 - lam, exp_coef)

    # Gaussian label mix
    elif "gaussian_dummy" == label_mix_type:
        y1 = adjusted_gaussian_func(lam.cpu(), gaussian_h1)
        y2 = adjusted_gaussian_func((1 - lam).cpu(), gaussian_h1)

    # Sine label mix
    elif "sine_dummy" == label_mix_type:
        y1 = sine_func(lam.cpu())
        y2 = sine_func((1 - lam).cpu())

    # Piecewise linear label mix
    elif "piecewise_linear_dummy" == label_mix_type:
        y1 = piecewise_func(lam.cpu(), piecewise_linear_h1, piecewise_linear_h2)
        y2 = piecewise_func((1 - lam).cpu(), piecewise_linear_h1, piecewise_linear_h2)

    else:
        raise AssertionError(f"There is no mix type: {label_mix_type}")

    y3 = (1 - y1 - y2)
    if zero_stack is not None:
        label = torch.hstack((label1, zero_stack)) * y1 + torch.hstack((label2, zero_stack)) * y2 + label3 * y3
    else:
        label = label1[:, :label3.size(1)] * y1 + label2[:, :label3.size(1)] * y2 + label3 * y3

    if not use_softlabel:
        label = label1[:, :label3.size(1)] * lam + label2[:, :label3.size(1)] * (1 - lam)

    return label, mix_label_mask


def mixup_process(out, target_reweighted, num_base_classes, lam, use_hard_positive_aug,
                  add_noise_level=0., mult_noise_level=0., hpa_type="none",
                  label_sharpening=True, label_mix="vanilla", label_mix_threshold=0.2, exp_coef=1.,
                  predefined_indices=None, gaussian_h1=0.2, piecewise_linear_h1=0.5, piecewise_linear_h2=0.):
    if predefined_indices is None:
        indices = np.random.permutation(out.size(0))
    else:
        indices = predefined_indices

    if use_hard_positive_aug:
        if not label_sharpening:  # turn off noise perturbation
            if hpa_type == "inter_class":
                out = out * lam + out[indices] * (1 - lam)
            elif hpa_type == "intra_class":
                indices = intra_class_permuted_indices(target_reweighted)
                out = out * lam + out[indices] * (1 - lam)
            else:
                raise AssertionError("Not implemented yet.")

            target_reweighted = label_mix_process(target_reweighted, target_reweighted[indices],
                                                  num_base_classes, lam, label_mix, label_mix_threshold, exp_coef,
                                                  gaussian_h1, piecewise_linear_h1, piecewise_linear_h2)
        else:
            if hpa_type == "inter_class":
                out = out * lam + feature_noise(out[indices], add_noise_level, mult_noise_level) * (1 - lam)
            elif hpa_type == "intra_class":
                indices = intra_class_permuted_indices(target_reweighted)
                # TODO. lam -> (1-lam), (1-lam) -> lam
                out = out * lam + feature_noise(out[indices], add_noise_level, mult_noise_level) * (1 - lam)
            else:
                raise AssertionError("Not implemented yet.")
    else:
        out = out * lam + out[indices] * (1 - lam)
        target_reweighted = label_mix_process(target_reweighted, target_reweighted[indices],
                                              num_base_classes, lam, label_mix, label_mix_threshold, exp_coef,
                                              gaussian_h1, piecewise_linear_h1, piecewise_linear_h2)

    # t1 = target.data.cpu().numpy()
    # t2 = target[indices].data.cpu().numpy()
    # print (np.sum(t1==t2))
    return out, target_reweighted


def middle_mixup_process(out, target_reweighted, num_base_classes, lam, use_hard_positive_aug,
                         add_noise_level=0., mult_noise_level=0., hpa_type="none",
                         label_sharpening=True, label_mix="vanilla", label_mix_threshold=0.2, exp_coef=1.,
                         predefined_indices=None, gaussian_h1=0.2, piecewise_linear_h1=0.5, piecewise_linear_h2=0., use_softlabel=True):
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target_reweighted, mix_label_mask = middle_label_mix_process(target_reweighted, target_reweighted[indices],
                                                                 num_base_classes, lam, label_mix,
                                                                 label_mix_threshold, exp_coef, gaussian_h1,
                                                                 piecewise_linear_h1, piecewise_linear_h2, use_softlabel)
    return out, target_reweighted, mix_label_mask

def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam



# https://github.com/erichson/NFM
def feature_noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    with torch.cuda.device(0):
        if add_noise_level > 0.0:
            add_noise = add_noise_level * np.random.beta(2, 5) * torch.cuda.FloatTensor(x.shape).normal_()
        if mult_noise_level > 0.0:
            mult_noise = mult_noise_level * np.random.beta(2, 5) * (
                    2 * torch.cuda.FloatTensor(x.shape).uniform_() - 1) + 1
    return mult_noise * x + add_noise
