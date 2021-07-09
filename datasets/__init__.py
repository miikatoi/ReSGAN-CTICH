from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from datasets.dataset import CTICHDataset


def path_collate(batch):
    paths = []
    for data in batch:
        paths.append(data.pop('path'))
    batch = default_collate(batch)
    batch['path'] = paths
    return batch


def create_dataloader(opt, phase):
    shuffle = (phase == 'train')
    
    if opt.dataset == 'CTICH':
        dataset = CTICHDataset(opt, phase=phase)
    else:
        raise NotImplementedError('Dataset [{}] not implemented'.format(opt.dataset))

    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle, collate_fn=path_collate)
    return loader, dataset


def get_args(parser, dataset_name):
    if dataset_name == 'CTICH':
        cmd_parser_modifier = CTICHDataset.modify_cmd_parser
    else:
        raise NotImplementedError('Dataset [{}] not implemented'.format(dataset_name))
    cmd_parser_modifier(parser)