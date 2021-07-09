import torch
import torch.nn as nn
from networks.generator import Generator
from networks.discriminator import NLayerDiscriminator, MultiscaleDiscriminator
from networks.segmentation import Unet
from networks.loss import VGGLoss, DFMLoss, SegLoss, GANLoss, KLDLoss


def create_generator(opt):
    net = Generator(opt)
    weights_init_normal(net)
    return net.to(opt.device)


def create_discriminator(opt):

    if opt.netD == 'NLayerDiscriminator':
        net = NLayerDiscriminator(input_nc=opt.input_nc, ndf=opt.ndf, n_layers=opt.D_downsamples, norm_layer=nn.BatchNorm2d)
    elif opt.netD == 'MultiscaleDiscriminator':
        net = MultiscaleDiscriminator(opt)
    else:
        raise NotImplementedError('netD [{}] not found'.format(opt.netD))

    weights_init_normal(net)

    return net.to(opt.device)


def create_segmentation(opt):
    net = Unet(1, output_nc=opt.semantic_nc, num_downs=3, norm_layer=nn.InstanceNorm2d)
    weights_init_normal(net)
    return net.to(opt.device)


def weights_init_normal(m, scale=0.02):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, scale)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, scale)
        torch.nn.init.constant_(m.bias.data, 0.0)
