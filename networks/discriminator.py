import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import torch.nn.utils.spectral_norm as spectral_norm


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, opt):
        super(MultiscaleDiscriminator, self).__init__()
        self.opt = opt

        for i in range(opt.n_discriminators):
            self.add_module('D_%d' % i, NLayerDiscriminator(input_nc=opt.input_nc, semantic_nc=opt.semantic_nc, ndf=opt.ndf, n_layers=opt.D_downsampling, norm_layer=nn.BatchNorm2d, D_mode=opt.D_mode))

    def downsample(self, x):
        return F.avg_pool2d(x, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    def forward(self, x, y=None, return_features=False):
        
        ds = []
        feats = []

        for _, net in self.named_children():
    
            if return_features:
                d, feat = net(x, y, return_features=return_features)
                ds.append(d)
                feats.append(feat)
            else:
                d = net(x, y)
                ds.append(d)

            x = self.downsample(x)
            if y is not None:
                y = self.downsample(y)
        
        if return_features:
            return ds, feats
        else:
            return ds


### Below code mostly borrowed from https://github.com/taesungp/contrastive-unpaired-translation ###


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, semantic_nc=0, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=True, D_mode='normal'):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.D_mode = D_mode

        if 'concat_label' in D_mode:
            input_nc += semantic_nc
        elif D_mode == 'normal':
            pass
        else:
            raise NotImplementedError('D mode [{}] not implemented'.format(D_mode))

        self.layers = []

        kw = 4
        padw = 1
        if(no_antialias):
            self.layers.append(nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)))
        else:
            self.layers.append(nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)))

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                self.layers.append(nn.Sequential(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ))

        self.layers.append(nn.Sequential(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)))  # output 1 channel prediction map
        self.model = nn.Sequential(*self.layers)

    def forward(self, x, y, return_features=False):
        """Standard forward."""
        if 'concat_label' in self.D_mode:
            x = torch.cat((x, y), axis=1)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        if return_features:
            return features[-1], features[:-1]
        else:
            return features[-1]


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
