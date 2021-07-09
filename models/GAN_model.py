import os
import torch
import torch.nn as nn   
import torch.nn.functional as F
import numpy as np

from torchvision.utils import save_image
#from apex import amp
from pathlib import Path
import shutil

from networks import VGGLoss, DFMLoss, GANLoss, SegLoss, KLDLoss
from networks import create_generator, create_discriminator, create_segmentation
from utils import scale_image_tensor, semantic_map_to_image
from .Base_model import BaseModel


class GANModel(BaseModel):

    @staticmethod
    def modify_cmd_parser(parser):
        parser = BaseModel.modify_cmd_parser(parser)

        parser.set_defaults(pretrain_net_names='')
        parser.set_defaults(train_net_names='G,D,S')
        parser.set_defaults(training_variable_names='loss_G,loss_D,loss_S,loss_G_GAN,loss_DFM,loss_VGG,loss_G_seg,loss_KLD')
        parser.set_defaults(validation_variable_names='')

        parser.add_argument("--lambda_vgg", type=float, default=10,
            help="VGG feature matching weight")
        parser.add_argument("--lambda_dfm", type=float, default=10,
            help="Discriminator feature matching weight")
        parser.add_argument("--lambda_G_seg", type=float, default=10,
            help="Segmentation match loss weight")
        parser.add_argument("--lambda_kld", type=float, default=0,
            help="KLD loss for VAE generator")

        parser.add_argument("--percep_mask", type=float, default=1,
            help="Mask perceptual loss")
        parser.add_argument("--gan_mode", type=str, default='hinge',
            help="gan loss type [lsgan | hinge]")

        parser.add_argument("--backtranslation", type=int, default=0,
            help="Whether to regenerate source map from fake image for additional training")
        
        parser.add_argument("--G_mode", type=str, default='bottleneck',
            help="Generator mode [bottleneck | skips | STU-skips | decoder]")
        parser.add_argument("--norm_G", type=str, default='sn', help="SPADE normalization")
        parser.add_argument("--skip_mode", type=str, default='bottleneck',
            help="Generator mode [bottleneck | skips | STU-skips | decoder]")
        parser.add_argument("--encoder_block_type", type=str, default='conv',
            help="Encoder block [conv | cond_conv | SPADE | CLADE]")
        parser.add_argument("--decoder_block_type", type=str, default='CLADE',
            help="Decoder block [conv | cond_conv | SPADE | CLADE]")
        parser.add_argument("--translation_block_type", type=str, default='CLADE',
            help="Translation block [conv | cond_conv | SPADE | CLADE]")
        parser.add_argument("--netD", type=str, default='MultiscaleDiscriminator', 
            help="The discriminator architecture to use [NLayerDiscriminator | MultiscaleDiscriminator]")
        parser.add_argument("--D_mode", type=str, default='concat_label',
            help="Discriminator mode [normal | concat_label]")
        parser.add_argument("--n_discriminators", type=int, default=2,
            help="If using multiscale discriminator, number of discriminators")
        parser.add_argument("--ngf", type=int, default=32,
            help="Number of generator root features")
        parser.add_argument("--ndf", type=int, default=64,
            help="Number of discriminator root features")
        parser.add_argument("--G_downsampling", type=int, default=4,
            help="Number of downsampling in generator")
        parser.add_argument("--D_downsampling", type=int, default=3,
            help="Number of downsampling in discriminator")
        parser.add_argument("--n_blocks", type=int, default=0,
            help="Number of blocks in translation network")

        return parser

    def __init__(self, opt, dataset):
        self.dataset = dataset
        super().__init__(opt)
        self.get_imgs(init=True)
         
    def set_input(self, batch, training=True):
        self.source_img = batch['source_img'].to(self.opt.device)
        self.source_label = batch['source_label'].to(self.opt.device)

    def create_networks(self):
        self.net_G = create_generator(self.opt)
        self.net_D = create_discriminator(self.opt)
        self.net_S = create_segmentation(self.opt)

    def create_losses(self):
        self.criterionGAN = GANLoss(self.opt.gan_mode).to(self.opt.device)
        self.criterionVGG = VGGLoss(self.opt).to(self.opt.device)
        self.criterionDFM = DFMLoss(self.opt).to(self.opt.device)
        w = torch.tensor(self.dataset.correcting_weights).to(self.opt.device)
        self.criterionSegS = SegLoss(weight=w)
        self.criterionSegG = SegLoss()
        self.criterionKLD = KLDLoss()
        
    def create_optimizers(self):
        self.optim_names = ['G', 'D', 'S']
        self.optim_G = torch.optim.Adam(self.net_G.parameters(), lr=self.opt.net_G_lr, betas=(self.opt.b1, self.opt.b2))
        self.optim_D = torch.optim.Adam(self.net_D.parameters(), lr=self.opt.net_D_lr, betas=(self.opt.b1, self.opt.b2))
        self.optim_S = torch.optim.Adam(self.net_S.parameters(), lr=self.opt.net_S_lr, betas=(self.opt.b1, self.opt.b2))

    def forward(self, identity=False):
        
        # modify target labels
        self.target_label = self.source_label.clone()
        if not identity:
            self.target_label = self.dataset.apply_editing_ops(self.target_label)
        # find which pixels were modified
        self.mask, _ = torch.max(torch.abs(self.target_label - self.source_label), dim=1, keepdim=False)
        self.mask = self.mask.unsqueeze(1)

        self.fake_img = self.net_G(self.source_img, self.target_label)

        if 'vae' in self.opt.G_mode:
            self.logvar = self.net_G.E.logvar
            self.mu = self.net_G.E.mu

        if self.opt.backtranslation:
            self.cycle_img = self.net_G(self.fake_img.detach(), self.source_label)

    def optimize_G(self):

        d_fake = self.net_D(self.fake_img, self.target_label)
        self.loss_G_GAN = self.criterionGAN(d_fake, True, False)
        if self.opt.backtranslation:
            d_cycle = self.net_D(self.cycle_img, self.source_label)
            self.loss_G_GAN += self.criterionGAN(d_cycle, True, False)
        
        if self.opt.lambda_vgg:
            self.loss_VGG = self.criterionVGG(self.fake_img, self.source_img, self.mask)
            if self.opt.backtranslation:
                self.loss_VGG += self.criterionVGG(self.cycle_img, self.source_img, None)
            self.loss_VGG *= self.opt.lambda_vgg
        else:
            self.loss_VGG = 0

        if self.opt.lambda_dfm:
            self.loss_DFM = self.criterionDFM(self.fake_img, self.target_label, self.source_img, self.source_label, self.mask, self.net_D)
            if self.opt.backtranslation:
                self.loss_DFM += self.criterionDFM(self.cycle_img, self.source_label, self.source_img, self.source_label, None, self.net_D)
            self.loss_DFM *= self.opt.lambda_dfm
        else:
            self.loss_DFM = 0
        
        if self.opt.lambda_G_seg:
            self.fake_seg = self.net_S(self.fake_img)
            self.loss_G_seg = self.criterionSegG(self.fake_seg, self.target_label.detach())
            if self.opt.backtranslation:
                self.cycle_seg = self.net_S(self.cycle_img)
                self.loss_G_seg += self.criterionSegG(self.cycle_seg, self.source_label.detach())
            self.loss_G_seg *= self.opt.lambda_G_seg
        else:
            self.loss_G_seg = 0

        if self.opt.lambda_kld:
            self.loss_KLD = self.criterionKLD(self.mu, self.logvar) * self.opt.lambda_kld
        else:
            self.loss_KLD = 0

        self.loss_G = self.loss_G_GAN + self.loss_VGG + self.loss_DFM + self.loss_G_seg + self.loss_KLD

        # mixed precision training
        if self.opt.amp == 1:
            with amp.scale_loss(self.loss_G, self.optim_G, loss_id=0) as scaled_loss:
                scaled_loss.backward(retain_graph=False)
        else:
            self.loss_G.backward(retain_graph=False)
        self.optim_G.step()
        
    def optimize_D(self):

        d_real = self.net_D(self.source_img, self.source_label)
        d_fake = self.net_D(self.fake_img.detach(), self.target_label)
        self.loss_D_GAN = self.criterionGAN(d_real, True, True) + self.criterionGAN(d_fake, False, True) 
        if self.opt.backtranslation:
            d_cycle = self.net_D(self.cycle_img.detach(), self.source_label)
            self.loss_D_GAN += self.criterionGAN(d_cycle, False, True)
        self.loss_D = self.loss_D_GAN
        
        # mixed precision training
        if self.opt.amp == 1:
            with amp.scale_loss(self.loss_D, self.optim_D, loss_id=1) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_D.backward()
        self.optim_D.step()

    def optimize_S(self):

        self.net_S.zero_grad()
        self.real_seg = self.net_S(self.source_img)
        self.loss_S = self.criterionSegS(self.real_seg, self.source_label)

        # mixed precision training
        if self.opt.amp == 1:
            with amp.scale_loss(self.loss_S, self.optim_S, loss_id=2) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_S.backward()
        self.optim_S.step()

    def optimize_parameters(self):
        self.net_G.zero_grad()
        self.forward()
        self.optimize_G()
        self.net_D.zero_grad()
        self.optimize_D()
        if 'S' in self.train_net_names:
            self.net_S.zero_grad()
            self.optimize_S()

    def get_imgs(self, init=False):

        if init:
            self.imgnames = [
                'source_img',
                'fake_img',
                'cycle_img',
                'residual_img',
                'all_ich',
                'no_ich',
                'source_label',
                'target_label',
                'fake_seg',
                'real_seg',
                'mask']
            for name in self.imgnames:
                setattr(self, name, torch.zeros((self.opt.crop_size, self.opt.crop_size)))

        imgs = [
            scale_image_tensor(self.source_img, to_image=True),
            scale_image_tensor(self.fake_img, to_image=True),
            scale_image_tensor(self.cycle_img, to_image=True),
            scale_image_tensor((self.source_img - self.fake_img).abs(), to_image=True),
            scale_image_tensor(self.all_ich, to_image=True),
            scale_image_tensor(self.no_ich, to_image=True),
            semantic_map_to_image(self.source_label),
            semantic_map_to_image(self.target_label),
            semantic_map_to_image(self.fake_seg),
            semantic_map_to_image(self.real_seg),
            self.mask]

        return dict(zip(self.imgnames, imgs))

    def train_model(self, train_loader, val_loader):
        for epoch_idx in range(1, self.opt.n_epochs + 1):
            for batch_idx, data in enumerate(train_loader):

                batches_done = (epoch_idx - 1) * len(train_loader) + batch_idx

                # Save model
                if batches_done % self.opt.save_interval == 0:
                    self.save_networks(state=dict(epoch_idx=epoch_idx, batch_idx=batch_idx))
                
                self.set_input(data)
                self.optimize_parameters()
                self.update_state()

                # Report progress
                if batches_done % self.opt.log_interval == 0:
                    message = "[Epoch {:d}/{:d}] [Batch {:d}/{:d}] ".format(
                        epoch_idx, self.opt.n_epochs, batch_idx, len(train_loader))
                    message += '['
                    for key, value in self.state.items():
                        message += ' {:s}: {:.3f} '.format(key, value)
                        self.writer.add_scalar(key, value, batches_done)
                    message += ']'
                    print(message)

                # visualize outputs
                if batches_done % self.opt.viz_interval == 0:
                    imgs = dict(
                        fake_img=scale_image_tensor(self.fake_img, to_image=True),
                        real_img=scale_image_tensor(self.source_img, to_image=True),
                        label=semantic_map_to_image(self.source_label),
                        target=semantic_map_to_image(self.target_label),
                        residual=scale_image_tensor((self.source_img - self.fake_img).abs(), to_image=True),
                        fake_seg=semantic_map_to_image(self.fake_seg),
                        real_seg=semantic_map_to_image(self.real_seg),
                        mask=self.mask,
                    )
                    self.upload_images(imgs, batches_done)

            self.update_learning_rate(epoch_idx)
            
    def run_experiment(self, train_loader, val_loader, test_loader):
        self.train_model(train_loader, val_loader)
