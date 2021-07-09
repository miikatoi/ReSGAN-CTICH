import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

def compute_latent_size(img_size, n_stages, factor=2):
    ''' Compute latent vector size assuming rectangular image.'''
    return img_size // (factor ** n_stages)


def compute_latent_length(channels, n_stages, factor=2, cap=4):
    ''' Compute latent vector lenght.'''
    return channels * factor ** min(n_stages, cap)


def create_network_block(block_type, in_nf, out_nf, cond_nf, kernel_size, stride,
            padding=0, norm='none', activation='relu', pad_type='zero', use_dropout=False):
    if 'conv' in block_type:
        if 'cond' not in block_type:
            cond_nf = 0
        block = ConvBlock(in_nf, out_nf, cond_nf, kernel_size, stride, padding, norm, activation, pad_type)
    elif 'resnet' in block_type:
        if 'cond' not in block_type:
            cond_nf = 0
        block = ResnetBlock(in_nf, out_nf, cond_nf, norm, pad_type, use_dropout)
    elif block_type == 'SPADE':
        block = SPADEResnetBlock(in_nf, out_nf, cond_nf, norm, 'spade')
    elif block_type == 'CLADE':
        block = SPADEResnetBlock(in_nf, out_nf, cond_nf, norm, 'clade')
    else:
        raise NotImplementedError('Network block [{}] not implemented'.format(block_type))
    
    return block


def create_encoder(opt):

    if opt.encoder_block_type in ['cond_conv', 'cond_resnet', 'CLADE', 'SPADE']:
        semantic_nc = opt.semantic_nc
    elif opt.encoder_block_type in ['conv', 'resnet']:
        semantic_nc = 0
    else:
        raise NotImplementedError('Encoder block [{}] not implemented.'.format(opt.encoder_block_type))
    return EncoderNetwork(
        semantic_nc=semantic_nc,
        nf=opt.ngf,
        G_downsampling=opt.G_downsampling,
        block_type=opt.encoder_block_type,
        G_mode=opt.G_mode,
        opt=opt)


def create_decoder(opt):

    if opt.decoder_block_type in ['cond_conv', 'cond_resnet', 'CLADE', 'SPADE']:
        semantic_nc = opt.semantic_nc
    elif opt.decoder_block_type in ['conv', 'resnet']:
        semantic_nc = 0
    else:
        raise NotImplementedError('Decoder block [{}] not implemented.'.format(opt.decoder_block_type))
    return DecoderNetwork(
        semantic_nc=semantic_nc,
        nf=opt.ngf,
        G_downsampling=opt.G_downsampling,
        block_type=opt.decoder_block_type,
        G_mode=opt.G_mode,
        opt=opt)


def create_translation(opt):

    if opt.translation_block_type in ['cond_conv', 'cond_resnet', 'CLADE', 'SPADE']:
        semantic_nc = opt.semantic_nc
    elif opt.translation_block_type in ['conv', 'resnet']:
        semantic_nc = 0
    else:
        raise NotImplementedError('Translation block [{}] not implemented.'.format(opt.translation_block_type))
    return TranslationNetwork(
        semantic_nc=semantic_nc,
        nf=compute_latent_length(opt.ngf, opt.G_downsampling),
        n_blocks=opt.n_blocks,
        block_type=opt.translation_block_type)


class EncoderNetwork(nn.Module):

    def __init__(self, semantic_nc, input_nc=1, nf=64, G_downsampling=3, block_type='conv', G_mode='bottleneck', norm='sn', activation='relu', opt=None):
        super(EncoderNetwork, self).__init__()

        self.G_mode = G_mode
        self.use_condition = (block_type in ['cond_conv', 'CLADE', 'SPADE'])
        self.latent_nf = compute_latent_length(nf, G_downsampling)    
        self.latent_hw = compute_latent_size(opt.crop_size, G_downsampling)

        model = [create_network_block('conv', input_nc, nf, 0, 7, 1, 3, norm=norm, activation=activation, pad_type='reflect')]
        
        # downsample n_ds times
        for ds_idx in range(G_downsampling):
            in_nf = compute_latent_length(nf, ds_idx, cap=4)
            out_nf = compute_latent_length(nf, ds_idx + 1, cap=4)
            model.append(create_network_block(block_type, in_nf, out_nf, semantic_nc, 3, 2, 1, norm=norm, activation=activation))

        if 'vae' in self.G_mode:
            self.fc_mu = nn.Linear(out_nf * self.latent_hw ** 2, self.latent_nf)
            self.fc_var = nn.Linear(out_nf * self.latent_hw ** 2, self.latent_nf)
            self.fc_z = nn.Linear(self.latent_nf, out_nf * self.latent_hw ** 2)

        self.model = nn.Sequential(*model)


    def forward(self, x, c=None):
        
        features = []
        for layer in self.model:
            x = layer(x, c)
            if 'skips' in self.G_mode:
                features.append(x)

        if 'vae' in self.G_mode:
            orig_shape = x.shape
            x = x.view(orig_shape[0], -1)
            self.mu = self.fc_mu(x)
            self.logvar = self.fc_var(x)
            std = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std) + self.mu
            z = self.fc_z(z)
            z = z.view(orig_shape)
            return z, features[1:-1]
            
        if 'skips' in self.G_mode:
            return x, features[1:-1]
        else:
            return x

    def encode(self, x, c=None):
        features = []
        for layer in self.model:
            x = layer(x, c)
            features += [x]
        return features[-3:]


class DecoderNetwork(nn.Module):

    def __init__(self, semantic_nc, output_nc=1, nf=64, G_downsampling=3, block_type='conv', G_mode='bottleneck', norm='sn', activation='relu', opt=None):
        super(DecoderNetwork, self).__init__()

        self.G_mode = G_mode
        self.opt = opt

        model = []
        merges = []

        # downsample n_ds times
        for ds_idx in reversed(range(G_downsampling)):
            in_nf = compute_latent_length(nf, ds_idx + 1)
            out_nf = compute_latent_length(nf, ds_idx)
            if 'cat_skips' in self.opt.skip_mode:
                merges.append(create_network_block('conv', 2 * out_nf, out_nf, 0, 1, 1, 0, norm=norm, activation=activation))
            model.append(create_network_block(block_type, in_nf, out_nf, semantic_nc, 3, 1, 1, norm=norm, activation=activation))

        model.append(create_network_block('conv', out_nf, output_nc, 0, 3, 1, 1, norm='none', activation='tanh'))

        self.up = nn.Upsample(scale_factor=2)
        self.model = nn.Sequential(*model)
        self.merges = nn.Sequential(*merges)

    def forward(self, x=None, skips=None, c=None):

        if skips is not None:
            skips = skips[::-1]

        
        for idx, layer in enumerate(self.model):
            x = layer(x, c)
            if idx < len(self.model) - 1:   # last layer needs no upsample
                x = self.up(x)
            if idx < 2:
                if 'sum_skips' in self.opt.skip_mode:
                    x += skips[idx]
                if 'cat_skips' in self.opt.skip_mode:
                    x = self.merges[idx](torch.cat((x, skips[idx]), axis=1))

        return x


class TranslationNetwork(nn.Module):

    def __init__(self, semantic_nc, nf=64, n_blocks=3, block_type='conv', norm='sn', activation='relu'):
        super(TranslationNetwork, self).__init__()

        model = []

        # downsample n_ds times
        for _ in range(n_blocks):
            model.append(create_network_block(block_type, nf, nf, semantic_nc, 3, 1, 1, norm=norm, activation=activation))

        self.model = nn.Sequential(*model)

    def forward(self, x, c):
        for block in self.model:
            x = block(x, c)
        return x


class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()

        self.opt = opt
        self.G_mode = opt.G_mode
        self.n_blocks = opt.n_blocks
        
        if self.G_mode in ['bottleneck', 'STU_skips', 'skips', 'vae', 'vae_skips']:
            self.E = create_encoder(opt)
            self.D = create_decoder(opt)
        elif self.G_mode in ['decoder']:
            self.D = create_decoder(opt)
        else:
            raise NotImplementedError('Generator mode [{}] not implemented'.format(self.G_mode))

        if self.n_blocks > 0:
            self.T = create_translation(opt)

    def forward(self, x, c=None):

        if 'skips' in self.G_mode:
            z, skips = self.E(x, c)
        elif 'bottleneck' in self.G_mode:
            z = self.E(x, c)
            skips = None
        elif 'vae' in self.G_mode:
            z, skips = self.E(x, c)
            if 'skips' not in self.G_mode:
                skips = None
        else:
            s = compute_latent_size(x.shape[2], self.opt.G_downsampling)
            l = compute_latent_length(self.opt.ngf, self.opt.G_downsampling)
            z = torch.rand((x.shape[0], l, s, s)).cuda()
            skips = None

        if self.n_blocks > 0:
            z = self.T(z, c)

        y = self.D(z, skips, c)
        
        return y


### Below code mostly borrowed from https://github.com/taesungp/contrastive-unpaired-translation and https://github.com/NVlabs/SPADE ###

class ConvBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim, cond_nf, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(ConvBlock, self).__init__()

        self.conditional = (cond_nf != 0)

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        use_bias = (norm != 'bn')
        self.conv = nn.Conv2d(input_dim + cond_nf, output_dim, kernel_size, stride, bias=use_bias)

        if norm == 'sn':
            self.conv = spectral_norm(self.conv)

    def forward(self, x, c=None):
        if self.conditional:
            c = F.interpolate(c, size=x.size()[2:], mode='nearest')
            x = torch.cat((x, c), axis=1)
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, out_dim, cond_nf, norm_type, padding_type, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.resize_channels = (dim != out_dim)
        self.build_conv_block(dim, out_dim, cond_nf, norm_type, padding_type, use_dropout)

    def build_conv_block(self, dim, out_dim, cond_nf, norm_type, padding_type, use_dropout):
        self.conv1 = ConvBlock(dim, dim, cond_nf, 3, 1, 1, norm=norm_type, activation='relu', pad_type=padding_type)
        self.conv2 = ConvBlock(dim, dim, 0, 3, 1, 1, norm=norm_type, activation='none', pad_type=padding_type)
        if self.resize_channels:
            self.conv3 = ConvBlock(dim, out_dim, 0, 1, 1, 0, norm=norm_type, activation='none', pad_type=padding_type)

    def forward(self, x, c=None):
        out = x + self.conv2(self.conv1(x, c))
        if self.resize_channels:
            out = self.conv3(out)
        return out

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc, norm_type, norm_block_type):
        super().__init__()

        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'sn' in norm_type:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        if norm_block_type == 'spade':
            self.norm_0 = SPADE(fin, semantic_nc)
            self.norm_1 = SPADE(fmiddle, semantic_nc)
            if self.learned_shortcut:
                self.norm_s = SPADE(fin, semantic_nc)
        elif norm_block_type == 'clade':
            self.norm_0 = SPADELight(fin, semantic_nc)
            self.norm_1 = SPADELight(fmiddle, semantic_nc)
            if self.learned_shortcut:
                self.norm_s = SPADELight(fin, semantic_nc)
        else:
            raise ValueError('%s is not a defined normalization method' % norm_block_type)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        param_free_norm_type = 'instance'
        ks = 3

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADELight(nn.Module):
    def __init__(self, norm_nc, label_nc, no_instance=True, add_dist=False):
        super().__init__()
        self.no_instance = no_instance
        self.add_dist = add_dist
        param_free_norm_type = 'instance'
        ks = 3

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        self.class_specified_affine = ClassAffine(label_nc, norm_nc, add_dist)

        if not no_instance:
            self.inst_conv = nn.Conv2d(1, 1, kernel_size=1, padding=0)

    def forward(self, x, segmap, input_dist=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. scale the segmentation mask
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        if not self.no_instance:
            inst_map = torch.unsqueeze(segmap[:,-1,:,:],1)
            segmap = segmap[:,:-1,:,:]

        # Part 3. class affine with noise
        out = self.class_specified_affine(normalized, segmap, input_dist)

        if not self.no_instance:
            inst_feat = self.inst_conv(inst_map)
            out = torch.cat((out, inst_feat), dim=1)

        return out


class ClassAffine(nn.Module):
    def __init__(self, label_nc, affine_nc, add_dist=False):
        super(ClassAffine, self).__init__()
        self.add_dist = add_dist
        self.affine_nc = affine_nc
        self.label_nc = label_nc
        self.weight = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        self.bias = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)
        if add_dist:
            self.dist_conv_w = nn.Conv2d(2, 1, kernel_size=1, padding=0)
            nn.init.zeros_(self.dist_conv_w.weight)
            nn.init.zeros_(self.dist_conv_w.bias)
            self.dist_conv_b = nn.Conv2d(2, 1, kernel_size=1, padding=0)
            nn.init.zeros_(self.dist_conv_b.weight)
            nn.init.zeros_(self.dist_conv_b.bias)

    def affine_gather(self, input, mask):
        n, c, h, w = input.shape
        # process mask
        mask2 = torch.argmax(mask, 1) # [n, h, w]
        mask2 = mask2.view(n, h*w).long() # [n, hw]
        mask2 = mask2.unsqueeze(1).expand(n, self.affine_nc, h*w) # [n, nc, hw]
        # process weights
        weight2 = torch.unsqueeze(self.weight, 2).expand(self.label_nc, self.affine_nc, h*w) # [cls, nc, hw]
        bias2 = torch.unsqueeze(self.bias, 2).expand(self.label_nc, self.affine_nc, h*w) # [cls, nc, hw]
        # torch gather function
        class_weight = torch.gather(weight2, 0, mask2).view(n, self.affine_nc, h, w)
        class_bias = torch.gather(bias2, 0, mask2).view(n, self.affine_nc, h, w)
        return class_weight, class_bias

    def forward(self, input, mask, input_dist=None):
        '''
        This is one way to guided sample the weights though einsum function
        # class_weight = torch.einsum('ic,nihw->nchw', self.weight, mask)
        # class_bias = torch.einsum('ic,nihw->nchw', self.bias, mask)
        '''
        class_weight, class_bias = self.affine_gather(input, mask)
        if self.add_dist:
            input_dist = F.interpolate(input_dist, size=input.size()[2:], mode='nearest')
            class_weight = class_weight * (1 + self.dist_conv_w(input_dist))
            class_bias = class_bias * (1 + self.dist_conv_b(input_dist))
        x = input * class_weight + class_bias
        return x
