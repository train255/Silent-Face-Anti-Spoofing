# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from src.data_io.dataset_folder import DatasetFolderFT
from src.data_io import transform as trans
import numpy as np

def get_train_loader(conf):
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4,
                          contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    dataset = DatasetFolderFT(root_path, train_transform,
                               None, conf.ft_width, conf.ft_height)

    # Creating data indices for training and validation splits:
    shuffle_dataset = True
    random_seed= 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(conf.val_size * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        sampler=train_sampler,
        num_workers=16
    )
    val_loader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        sampler=valid_sampler,
        num_workers=16
    )

    return train_loader, val_loader
