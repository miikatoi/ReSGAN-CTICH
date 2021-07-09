import argparse
import torch
import os
from pathlib import Path

import datasets
import models


def create_options():

    parser = argparse.ArgumentParser()
    base = parser.add_argument_group("base")
    model_group = parser.add_argument_group("model")
    dataset_group = parser.add_argument_group("dataset")

    # identification
    base.add_argument("--name", type=str, default='untitled', help="name of the training run")

    # training data
    base.add_argument("--dataroot", type=str, help="Dataset path")
    base.add_argument("--dataset", type=str, help="Dataset name")
    base.add_argument("--img_size", type=int, default=540, help="size of the images to load")
    base.add_argument("--crop_size", type=int, default=512, help="randomly crop images to this size")
    base.add_argument("--input_nc", type=int, default=1, help="Number of input channels. Must match encoder model")
    base.add_argument("--preprocess", type=str, default='resize+vflip+hflip+rcrop', help="Include ops to string: [resize, vflip, hflip, rcrop]")

    # training
    base.add_argument("--batch_size", type=int, default=8, help="size of the images to load")
    base.add_argument("--n_epochs", type=int, default=200, help="number of epochs to train for")
    base.add_argument("--b1", type=float, default=0, help="Adam optimizer parameter beta 1")
    base.add_argument("--b2", type=float, default=0.999, help="Adam optimizer parameter beta 2")
    base.add_argument("--net_G_lr", type=float, default=0.0001, help="Generator learning rate")
    base.add_argument("--net_D_lr", type=float, default=0.0004, help="Discriminator learning rate")
    base.add_argument("--net_S_lr", type=float, default=0.0002, help="Semantic critic learning rate")
    base.add_argument("--lr_scheduler", type=str, default='lineardecay', help="Learning rate scheduler [none | lineardecay]")
    base.add_argument("--decay_start_epoch", type=int, default=100, help="Which epoch to start decay from")

    # environment
    base.add_argument("--cuda", type=int, default=1, help="Use GPU")
    base.add_argument("--amp", type=int, default=0, help="Use mixed precision training")    # uncomment apex from GAN_Model.py if you want to use
    base.add_argument("--log_interval", type=int, default=10, help="log every N batches")
    base.add_argument("--viz_interval", type=int, default=260, help="upload images every N batches")
    base.add_argument("--save_interval", type=int, default=260, help="Save model every N batches")
    base.add_argument("--outputs_dir", type=str, default='outputs', help="Directory for saving models")
    base.add_argument("--phase", type=str, default='train', help="train/val/test")

    # gather dataset and model specific options
    opt, _ = parser.parse_known_args()
    datasets.get_args(dataset_group, opt.dataset)
    models.get_args(model_group)

    opt = parser.parse_args()

    if not torch.cuda.is_available():
        opt.cuda = False
        opt.device = 'cpu'
    else:
        if opt.cuda:
            opt.device = 'cuda'

    # modify paths
    opt.outputs_dir = os.path.join(opt.outputs_dir, opt.name)
    # Create output directories
    Path(opt.outputs_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(opt.outputs_dir, 'options.txt'), mode='w') as f:
        f.write(str(opt))

    # check arguments
    if opt.name == 'untitled':
        print('Warning: No name given to the experiment, saving as [{}]'.format(opt.name))  # warning

    opt.eval_methods = opt.eval_methods.split(',')

    # print options groupwise
    ns = 35
    msg = '#' * 2 * ns
    msg += '\nCreated options:\n'
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(opt, a.dest, None) for a in group._group_actions}
        msg += '\n[{}]\n'.format(group.title)
        opts = argparse.Namespace(**group_dict)
        for arg in vars(opts):
            msg += '{1:<{0}}{2}\n'.format(ns, arg, str(getattr(opts, arg)))
    msg += '#' * 2 * ns
    print(msg)


    return opt
