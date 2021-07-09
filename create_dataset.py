from datasets import create_dataloader
from options import create_options
from models import create_model

import os
from pathlib import Path
import torch


def generate(dataset):

    for batch_idx, data in enumerate(dataset):

        print('[{} / {}]'.format(batch_idx + 1, len(dataset)))

        model.eval()
        with torch.no_grad():
            model.set_input(data, training=False)
            model.forward()
            data['fake_img'] = model.fake_img.cpu()

        for idx, path in enumerate(data['path']):
            if opt.dataset == 'CTICH':
                data['path'][idx] = path.replace("image", "synthetic")
            elif opt.dataset == 'UH':
                data['path'][idx] = path.replace("ct", "synthetic")
            else:
                raise NotImplementedError('Unknown dataset {}'.format(opt.dataset))
            # create output dir
            Path(os.path.split(data['path'][idx])[0]).mkdir(parents=True, exist_ok=True)

        ds.save_images(data)

        data['fake_img'] = (model.source_img - model.fake_img).cpu()
        for idx, path in enumerate(data['path']):
            if opt.dataset == 'CTICH':
                data['path'][idx] = path.replace("synthetic", "residual")
            elif opt.dataset == 'UH':
                data['path'][idx] = path.replace("synthetic", "residual")
            else:
                raise NotImplementedError('Unknown dataset {}'.format(opt.dataset))
            # create output dir
            Path(os.path.split(data['path'][idx])[0]).mkdir(parents=True, exist_ok=True)

        ds.save_images(data)


opt = create_options()

opt.phase = 'test'
opt.preprocess = 'resize'
opt.img_size = opt.crop_size
opt.test_editing_ops = 'rmich'
opt.batch_size = 1

training_data, ds = create_dataloader(opt, 'train synthesis')
validation_data, ds = create_dataloader(opt, 'val synthesis')
testing_data, ds = create_dataloader(opt, 'test synthesis')

model = create_model(opt, ds)

print('Generating Training Set..')
generate(training_data)
print('Generating Validation Set..')
generate(validation_data)
print('Generating Test Set..')
generate(testing_data)
