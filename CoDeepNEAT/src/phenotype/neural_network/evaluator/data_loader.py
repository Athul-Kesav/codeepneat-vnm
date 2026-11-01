from os import path
from typing import List
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder, CIFAR100, FashionMNIST

from data import DataManager

from configuration import config
from src.phenotype.augmentations.augmentation_scheme import AugmentationScheme

_DL_CONTEXT = None


def _loader_context():
    global _DL_CONTEXT
    if _DL_CONTEXT is None:
        _DL_CONTEXT = mp.get_context("forkserver")
    return _DL_CONTEXT


def load_data(composed_transforms: transforms.Compose, split: str) -> DataLoader:
    """
    Loads the data given the config.dataset

    :param composed_transforms: the augmentations to apply to the dataset
    :param split: either train or test, dataset returned also depends on config.fully_train -
    'train' will return 42500 images for evolution, but 50000 for fully training.
    'test' will return 7500 images for evolution, but 10000 for fully training.

    Note: the validation/train split does not try balance the classes of data, it just takes the first n for the train
    set and the remaining data goes to the validation set
    """
    if split.lower() not in ['train', 'test']:
        raise ValueError('Parameter split can be one of train, test or validation, but received: ' + str(split))

    # when to load train set
    train: bool = True if split == 'train' or (split == 'test' and not config.fully_train) else False

    dataset_args = {
        'root': DataManager.get_datasets_folder(),
        'train': train,
        'download': config.download_dataset,
        'transform': composed_transforms
    }

    if config.dataset == 'mnist':
        dataset = MNIST(**dataset_args)
    elif config.dataset == 'cifar10':
        dataset = CIFAR10(**dataset_args)
    elif config.dataset == 'cifar100':
        dataset = CIFAR100(**dataset_args)
    elif config.dataset == 'fashionMnist':
        dataset = FashionMNIST(**dataset_args)
    elif config.dataset == 'custom':
        dataset = get_generic_dataset(composed_transforms, train)
    else:
        raise ValueError('config.dataset can be one of mnist, cifar10 or custom, but received: ' + str(config.dataset))

    if train and not config.fully_train:
        # ImageFolder stores classes contiguously, so a prefix split puts
        # whole classes only in train or only in validation.
        n = len(dataset)
        generator = torch.Generator().manual_seed(0)
        perm = torch.randperm(n, generator=generator).tolist()
        train_size = int(n * (1 - config.validation_split))
        if split == 'train':
            dataset = Subset(dataset, perm[:train_size])
        elif split == 'test':
            dataset = Subset(dataset, perm[train_size:])
        else:
            raise Exception("")

    # print(split, 'set size in', 'FT' if config.fully_train else 'evo', len(dataset))

    workers = 4 if config.image_size and config.image_size > 0 else 0
    loader_kwargs = dict(batch_size=config.batch_size, shuffle=True, num_workers=workers, pin_memory=workers > 0)
    if workers:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["multiprocessing_context"] = _loader_context()
    return DataLoader(dataset, **loader_kwargs)


def get_data_shape() -> List[int]:
    return list(next(iter(load_data(load_transform(), 'test')))[0].size())


def get_generic_dataset(composed_transforms: transforms.Compose, train: bool) -> Dataset:
    """
    Loads data from custom_dataset_root, given that the data is structured as follows:
    root/train/dog/xxx.png
    root/test/dog/xxy.png

    root/train/cat/123.png
    root/test/cat/nsdf3.png

    Note: the train set should contain enough data to be split into a train and validation set
    :return: a train and test dataloader
    """
    if train:
        data = ImageFolder(root=path.join(config.custom_dataset_root, 'train'), transform=composed_transforms)
    else:
        data = ImageFolder(root=path.join(config.custom_dataset_root, 'test'), transform=composed_transforms)

    return data


def load_transform(aug: AugmentationScheme = None) -> transforms.Compose:
    norm_tuple = (0.5, 0.5, 0.5) if config.use_colour_augmentations else (0.5,)

    if config.evolve_da and not config.batch_augmentation and aug is not None:
        return transforms.Compose([
            aug,
            transforms.ToTensor(),
            transforms.Normalize(norm_tuple, norm_tuple)
        ])
    else:
        ops = []
        if config.image_size and config.image_size > 0:
            side = int(config.image_size)
            ops.extend([transforms.Resize(side), transforms.CenterCrop(side)])
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(norm_tuple, norm_tuple)
        ])
        return transforms.Compose(ops)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
