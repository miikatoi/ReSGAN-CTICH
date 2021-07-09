
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import scale_image_tensor

### Below code mostly adapted from https://github.com/NVlabs/SPADE and https://github.com/taesungp/contrastive-unpaired-translation ###

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 24):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(24, 31):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):

        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)

        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)    
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                         
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, opt):

        super(VGGLoss, self).__init__()
        self.vgg = Vgg16().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0]
        self.opt = opt

    def forward(self, x, y, mask=None):

        # rescale to right input range
        x, y = scale_image_tensor(x, to_image=True), scale_image_tensor(y, to_image=True)

        if self.opt.percep_mask and mask is not None:
            x, y = x * (1 - mask), y * (1 - mask)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(self.weights)):
            # compute loss
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])   

        return loss


class DFMLoss(nn.Module):
    def __init__(self, opt):
        
        super(DFMLoss, self).__init__()        
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.opt = opt

    def forward(self, pred, target_label, source, source_label, mask, net):

        if self.opt.percep_mask and mask is not None:
            pred, source = scale_image_tensor(pred, to_image=True), scale_image_tensor(source, to_image=True)
            pred, source = pred * (1 - mask), source * (1 - mask)
            pred, source = scale_image_tensor(pred, to_image=False), scale_image_tensor(source, to_image=False)

        _, feat_x = net(pred, target_label, return_features=True)
        _, feat_y = net(source, source_label, return_features=True)

        start_idx = len(feat_x) - len(self.weights)
        feat_x, feat_y = feat_x[start_idx:], feat_y[start_idx:]            
        loss = 0

        if not isinstance(feat_x, list):
            feat_x, feat_y = [feat_x], [feat_y]

        for x, y in zip(feat_x, feat_y):

            for i in range(len(self.weights)):
                loss += self.weights[i] * self.criterion(x[i], y[i].detach())   

        return loss


class SegLoss(nn.Module):
    def __init__(self, weight=None):
        super(SegLoss, self).__init__()        
        if weight is not None:
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        gt = label.argmax(axis=1)
        #x = scale_image_tensor(pred, True) # only used if semantic critic (-1, 1) with tanh
        x = pred
        loss = self.criterion(x, gt)
        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('zero_tensor', torch.tensor(0))
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating', 'hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def __call__(self, input, target_is_real, for_discriminator=None):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        loss = 0
        
        # for handling multi-scale D
        if not isinstance(input, list):
            input = [input]

        for prediction in input:
            bs = prediction.size(0)
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss += self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss += -prediction.mean()
                else:
                    loss += prediction.mean()
            elif self.gan_mode == 'nonsaturating':
                if target_is_real:
                    loss += F.softplus(-prediction).view(bs, -1).mean(dim=1)
                else:
                    loss += F.softplus(prediction).view(bs, -1).mean(dim=1)
            elif self.gan_mode == 'hinge':
                if for_discriminator is None:
                    raise NotImplementedError('hinge loss used without specifying D/G on call')
                if for_discriminator:
                    if target_is_real:
                        minval = torch.min(prediction - 1, self.get_zero_tensor(prediction))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-prediction - 1, self.get_zero_tensor(prediction))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(prediction)
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())