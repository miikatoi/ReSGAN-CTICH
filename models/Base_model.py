import torch
import os
from torch.optim import lr_scheduler
#from apex import amp
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from utils import arg_to_list


class BaseModel(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.pretrain_net_names = arg_to_list(opt.pretrain_net_names)
        self.train_net_names = arg_to_list(opt.train_net_names)
        self.net_names = self.train_net_names + self.pretrain_net_names
        self.training_variable_names = arg_to_list(opt.training_variable_names)
        self.validation_variable_names = arg_to_list(opt.validation_variable_names)
        self.variable_names = self.training_variable_names + self.validation_variable_names

        self.create_networks()
        self.print_networks()
        if self.opt.phase == 'train':
            self.create_losses()
            self.create_optimizers()
            self.create_schedulers()
            if self.opt.amp:
                self.init_amp()
            self.writer = SummaryWriter()

        if self.opt.phase == 'train':
            self.load_networks(self.pretrain_net_names, self.opt.net_names_suffix)
        elif self.opt.phase == 'test':
            self.pretrain_net_names = self.net_names
            self.load_networks(self.net_names, self.opt.net_names_suffix)
        else:
            raise NotImplementedError('Unknown phase {}'.format(self.opt.phase))

        # push all variables in here and use lists to pick which to upload
        self.state = dict.fromkeys(self.variable_names, 0)
            
    @staticmethod
    def modify_cmd_parser(parser):
        parser.add_argument("--eval_methods", type=str, default='',
            help="Which methods to use for evaluation [mIoU | Dice | FID]")
        parser.add_argument("--pretrain_net_names", type=str,
            help="Names of pretrained networks separated by comma [G | S | D]")
        parser.add_argument("--train_net_names", type=str,
            help="Names of networks being trained separated by comma [G | S | D]")
        parser.add_argument("--training_variable_names", type=str,
            help="Names of training variables to log separated by comma [lossname]")
        parser.add_argument("--validation_variable_names", type=str,
            help="Names of validation variables to log separated by comma [FID | mIoU | Dice]")
        parser.add_argument("--net_names_suffix", type=str, default=None,
            help="Suffix of net names, 'best_acc' etc.]")
        
        return parser

    def print_networks(self):
        net_instances = self.get_instances(self.net_names, 'net')
        for net, name in zip(net_instances, self.net_names):
            msg = ''
            if name in self.pretrain_net_names:
                msg += 'pretrained '
            msg += 'network {:s} created'
            print(msg)

    def print_optimizers(self):
        optim_instances = self.get_instances(self.optim_names, 'optim')
        for optim, name in zip(optim_instances, self.optim_names):
            print('{} optimizer optim_{:s} created'.format(optim.__class__.__name__, name))

    def load_networks(self, nets=None, suffix=None):
        if nets is None:
            nets = self.pretrain_net_names
        net_instances = self.get_instances(nets, 'net')
        for net, name in zip(net_instances, self.pretrain_net_names):
            checkpoint = torch.load(self.get_checkpoint_path(name, suffix))
            net.load_state_dict(checkpoint['state_dict'])
            if 'state' in checkpoint.keys():
                return checkpoint['state']

    def save_networks(self, nets=None, suffix=None, state=None):
        if nets is None:
            nets = self.train_net_names
        net_instances = self.get_instances(nets, 'net')
        for net, name in zip(net_instances, nets):
            checkpoint = dict(state_dict=net.state_dict())
            if state is not None:
                checkpoint['state'] = state
            torch.save(checkpoint, self.get_checkpoint_path(name, suffix))

    def update_learning_rate(self, epoch):
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step(epoch) # epoch not required?
        optim_instances = self.get_instances(self.optim_names, 'optim')
        for optimizer, name in zip(optim_instances, self.optim_names):
            lr = optimizer.param_groups[0]['lr']
            print('Optimizer [optimizer{:s}] learning rate = {:.7f}'.format(name, lr))

    def create_schedulers(self):
        optim_instances = self.get_instances(self.optim_names, 'optim')
        self.schedulers = [self.create_scheduler(optimizer) for optimizer in optim_instances]

    def create_scheduler(self, optimizer):

        def lambda_rule(epoch):
            decay_lenght = self.opt.n_epochs - self.opt.decay_start_epoch
            decay_start = self.opt.decay_start_epoch
            lr_l = 1.0 - max(0, epoch - decay_start) / float(decay_lenght + 1)
            return lr_l
        
        if self.opt.lr_scheduler == 'lineardecay':
            return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif self.opt.lr_scheduler == 'none':
            return None
        else:
            raise NotImplementedError('LR scheduler [{}] not implemented'.format(self.opt.lr_scheduler))

    def init_amp(self):
        # uncomment to use amp
        pass
        '''# initialize mixed precision training
        models = [getattr(self, 'net_' + name) for name in self.train_net_names]
        optims = [getattr(self, 'optim_' + name) for name in self.train_net_names]
        models, optims = amp.initialize(models, optims, opt_level="O1", num_losses=len(optims))
        [setattr(self, 'net_' + name, net) for name, net in zip(self.train_net_names, models)]
        [setattr(self, 'optim_' + name, optim) for name, optim in zip(self.train_net_names, optims)]
        print('Mixed precision training with {:d} losses initialized'.format(len(optims)))'''

    def update_state(self):
        for name in self.variable_names:
            self.state[name] = getattr(self, name, -1)

    def get_instances(self, values, typestr=None):
        instances = []
        if not isinstance(values, list):
            values = [values]
        for value in values:
            if isinstance(value, str):
                if typestr is not None:
                    value = typestr + '_' + value
                else:
                    raise Exception('Typestring not given when getting attribute {}.'.format(value))
                instances.append(getattr(self, value))
        return instances

    def get_checkpoint_path(self, name, suffix=None):
        if suffix is not None:
            name = '{}_{}'.format(name, suffix)
        return os.path.join(self.opt.outputs_dir, 'net_{}.pth'.format(name))

    def upload_images(self, images, step=None):
        for name, img in images.items():
            img = make_grid(img)
            self.writer.add_image(name, img, step)
