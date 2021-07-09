import torch

from datasets import create_dataloader
from options import create_options

opt = create_options()
opt.phase = 'train'
training_data, training_dataset = create_dataloader(opt, 'train')
 
class_totals = torch.zeros((opt.semantic_nc))
for batch_idx, data in enumerate(training_data):
    class_totals += data['source_label'].sum(axis=[0,2,3])

class_stats = class_totals / class_totals.sum()

print('Proportions of classes: ', class_stats, '\ncorrecting weights: ', 1/class_stats/opt.semantic_nc)
