import torch
import numpy as np
from torchvision.utils import save_image


def param_count(net):
    n_params = sum([p.numel() for p in net.parameters()])
    return n_params


def arg_to_list(arg, dtype=str):

    ignore = ' '
    lst = list(filter(lambda x: x not in ignore, arg.split(',')))
    clear = '()[]{\}'
    lst = [''.join(list(filter(lambda x: x not in clear, x))) for x in lst]
    # convert to desired numeric datatype
    if dtype != str:
        lst = [dtype(x) for x in lst]
    return lst


def scale_image_tensor(x, to_image=False):
    if to_image:    # in range [-1, 1] to [0, 1] 
        return (x + 1) / 2
    else:           # in range [0, 1] to [-1, 1]
        return x * 2 - 1


def semantic_map_to_image(x):
    '''Maps semantic maps to colors. Assuming max values to be 1 and others 0'''
    x = binarize_tensor(x)
    tgt = torch.zeros(x.shape[0], 3, *x.shape[2:])
    rng = np.random.RandomState(987654321)
    clrs = torch.tensor([[rng.rand(), rng.rand(), rng.rand()] for _ in range(x.shape[1])])
    for img_idx in range(x.shape[0]):
        for ch_idx in range(tgt.shape[1]):
            cls_idx = torch.argmax(x[img_idx], dim=0)
            tgt[img_idx, ch_idx] = clrs[cls_idx, ch_idx]
    return tgt


def binarize_tensor(x):
    return (x > 0.5).float()

