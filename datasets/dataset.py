import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.segmentation import flood_fill
from PIL import Image
import numpy as np
import torch
import os

from utils import scale_image_tensor, save_image, arg_to_list


class BaseDataset(Dataset):

    def __init__(self, opt, phase):
        self.opt = opt
        self.dataroot = self.opt.dataroot
        self.phase = phase
        
        if 'range' in self.opt.slices:
            txt = self.opt.slices.replace('range','')
            limits = arg_to_list(txt, int)
            self.slices = range(*limits)   # inclusive range
        elif 'list' in self.opt.slices:
            txt = self.opt.slices.replace('list','')
            self.slices = arg_to_list(txt, int)
        else:
            raise NotImplementedError('Unrecognized slice selection method [{}]'.format(self.opt.slices))

        self.rmich_channels = arg_to_list(opt.rmich_channels, int)
        self.correcting_weights = arg_to_list(opt.semantic_channel_weights, float)

    @staticmethod
    def modify_cmd_parser(parser):
        parser.add_argument("--semantic_nc", type=int, default=3,
            help="Number of classes or channels")
        parser.add_argument("--rmich_channels", type=str,
            help="rmich for this dataset is swap a to b : (a,b)")
        parser.add_argument("--semantic_channel_weights", type=str,
            help="inbalance fixing weights to use for CE loss")
        parser.add_argument("--train_editing_ops", type=str, default='none,rmich',
            help="How to modify the semantic maps for training [none | rmich]")
        parser.add_argument("--test_editing_ops", type=str, default='none',
            help="How to modify the semantic maps for testing [none | rmich]")
        parser.add_argument("--slices", type=str, default='list(8)',
            help="Which slices to include in the data [list(0,1,...) | range(1,3)]")
        parser.add_argument("--data_limit", type=float, default=1,
            help="Proportion to limit the size of the dataset [0, 1]")

        return parser

    def swap_channel(self, tensor, idx_a, idx_b):
    
        tensor[idx_b] = torch.logical_or(tensor[idx_a], tensor[idx_b])
        tensor[idx_a] = torch.zeros(tensor[idx_b].shape).to(self.opt.device)
        return tensor

    def flip_channel(self, tensor, idx_a, idx_b):
        b = tensor[idx_b].clone()
        tensor[idx_b] = tensor[idx_a]
        tensor[idx_a] = b
        return tensor

    def sample_noise(self, tensor, idx_a, idx_b):
        # replace ch b elements with ch a noise

        x = torch.rand((1, 1, 16, 16))
        # x = F.interpolate(x, scale_factor=8, mode='bicubic')
        x = F.interpolate(x, size=tensor.shape[-2:], mode='bicubic')
        x = (x > 0.85).squeeze().cuda()

        x[tensor[idx_b] == 0] = 0
        tensor[idx_a][x == 1] = 1
        tensor[idx_b][x == 1] = 0
        return tensor


    def apply_editing_ops(self, labels, phase=None, ops=None):
        
        islist = isinstance(labels, list)

        if not islist:
            labels = [labels]

        if phase is None:   # phase is not overridden
            phase = self.phase

        if ops is None:
            if phase == 'train':
                ops = self.opt.train_editing_ops
            elif phase in ['val', 'test']:
                ops = self.opt.test_editing_ops
            elif 'synthesis' in phase:
                ops = self.opt.test_editing_ops
            else:
                raise NotImplementedError('Unrecognized training phase [{}]'.format(phase))
            
        ops = ops.strip().split(',')
        transforms = np.random.choice(ops, labels[0].shape[0])

        for label in labels:
            for idx, (tensor, tf) in enumerate(zip(label, transforms)):
                if tf == 'rmich':
                    tensor = self.swap_channel(tensor, *self.rmich_channels)
                elif tf == 'mkich':
                    tensor = self.sample_noise(tensor, *self.rmich_channels)
                elif tf == 'none':
                    pass
                else:
                    raise NotImplementedError('Transform [{}] not implementd'.format(tf))

        if not islist:
            labels = labels[0]
        return labels

    def save_images(self, data):

        fake = scale_image_tensor(data['fake_img'], to_image=True)

        for idx in range(fake.size(0)):
            save_image(fake[idx], data['path'][idx])

    def apply_transforms(self, imgs, label):
        '''Same opeartions for source and label'''
        islist = isinstance(imgs, list)
        
        if not islist:
            imgs = [imgs]

        # Resize
        if 'resize' in self.opt.preprocess:
            resize = transforms.Resize(size=(self.opt.img_size, self.opt.img_size))
            for idx in range(len(imgs)):
                imgs[idx] = resize(imgs[idx])
            label = [resize(x) for x in label]

        # augmentations
        if 'hflip' in self.opt.preprocess:
            if np.random.random() > 0.5:
                for idx in range(len(imgs)):
                    imgs[idx] = TF.hflip(imgs[idx])
                label = [TF.hflip(x) for x in label]
        if 'vflip' in self.opt.preprocess:
            if np.random.random() > 0.5:
                for idx in range(len(imgs)):
                    imgs[idx] = TF.vflip(imgs[idx])
                label = [TF.vflip(x) for x in label]
        if 'rcrop' in self.opt.preprocess:
            crop_indices = transforms.RandomCrop.get_params(imgs[0], output_size=(self.opt.crop_size, self.opt.crop_size))
            for idx in range(len(imgs)):
                imgs[idx] = TF.crop(imgs[idx], *crop_indices)
            label = [TF.crop(x, *crop_indices) for x in label]

        # To tensor
        for idx in range(len(imgs)):
            imgs[idx] = TF.to_tensor(imgs[idx])
        label = [TF.to_tensor(x) for x in label]

        # scale to nn output range -1 to 1, labels are 0 to 1
        for idx in range(len(imgs)):
            imgs[idx] = scale_image_tensor(imgs[idx])
        label = torch.cat(label, axis=0)

        if not islist:
            imgs = imgs[0]
        return imgs, label


class CTICHDataset(BaseDataset):

    @staticmethod
    def modify_cmd_parser(parser):
        parser = BaseDataset.modify_cmd_parser(parser)
        parser.set_defaults(semantic_nc=4)
        parser.set_defaults(slices='range0,100')
        parser.set_defaults(rmich_channels='3,2')
        parser.set_defaults(semantic_channel_weights='1.0,1.0,1.0,30.0')
        parser.add_argument("--bg_tresh", type=float, default=0.01 * 255,
            help="Pixel values below this are background")
        parser.add_argument("--bone_tresh", type=float, default=0.99 * 255,
            help="Pixel values above this are bone)")
        parser.add_argument("--datalist_path", type=str,
            help="Data list path.")
        parser.add_argument("--cv_index", type=int, default=5,
            help="Cross validation index.")
        return parser

    def __init__(self, opt, phase):
        super(CTICHDataset, self).__init__(opt, phase)

        # select the appropriate datalist
        if 'train' in phase:
            datalist_path = os.path.join(self.opt.datalist_path, 'splits/cv{:d}_train_split.lst'.format(self.opt.cv_index))
        elif 'val' in phase:
            datalist_path = os.path.join(self.opt.datalist_path, 'splits/cv{:d}_val_split.lst'.format(self.opt.cv_index))
        elif 'test' in phase:
            datalist_path = os.path.join(self.opt.datalist_path, 'splits/test_split.lst')
        else:
            raise NotImplementedError('Unrecognized training phase [{}]'.format(phase))

        # Read list of data paths from a file
        paths = np.array([line.strip().split() for line in open(datalist_path)])
        slices = [int(row[-1]) for row in paths]

        # Filter out unwanted slices
        selector = [idx in self.slices for idx in slices]
        self.img_paths = paths[selector]

        # uncomment to filter out non-ich slices
        '''if 'train' in phase:
            selector = []
            for path in self.img_paths:
                mask = Image.open(os.path.join(self.opt.dataroot, path[1])).convert('L')
                selector.append(np.array(mask).max() != 0)
            selector = np.array(selector)
            self.img_paths = self.img_paths[selector]'''

        if 'train' in phase:
            self.img_paths = self.img_paths[:int(self.img_paths.shape[0] * self.opt.data_limit)]

        print('Dataset [{:s} {:s}] created, {:d} samples from slices {:s}'.format(
            self.opt.dataset, phase, self.img_paths.shape[0], str(self.slices)))
            
    
    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, idx):

        # load image and label
        img_path = self.img_paths[idx]
        
        mask = Image.open(os.path.join(self.opt.dataroot, img_path[1])).convert('L')

        imgs = Image.open(os.path.join(self.opt.dataroot, img_path[0])).convert('L')
        img_path = os.path.join(self.opt.dataroot, img_path[0])

        # background, brain, hemorrhage
        ich = np.array(np.array(mask)) != 0
        bg = (np.logical_and(np.array(np.array(imgs)) == 0, ~ich)).astype(int)
        #filled_bg = flood_fill(bg, (0, 0), 2)  # floodfill is problematic without skullstripping, hence not used in this version
        #bg = (np.array(filled_bg) == 2).astype(int)
        bone = (np.array(np.array(imgs)) > self.opt.bone_tresh).astype(int)
        brain = (np.logical_and(~bone, np.logical_and(~bg, ~ich))).astype(int)
        semantic_label = [Image.fromarray(np.uint8(x)).convert('L') for x in [bg, bone, brain, ich]]

        # apply transforms
        imgs, semantic_label = self.apply_transforms(imgs, semantic_label)
        semantic_label = F.one_hot(semantic_label.argmax(0), num_classes=self.opt.semantic_nc).float().permute(2, 0, 1)

        img_path = os.path.join(self.opt.dataroot, img_path)

        return {'source_img': imgs, 'source_label': semantic_label, 'path': img_path}

