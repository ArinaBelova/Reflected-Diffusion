"""

author: Maximilian Springenberg
association: Fraunhofer HHI
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms.functional import to_tensor, center_crop
from torchvision.datasets.folder import default_loader


SFXS = '.jpg .jpeg .png .tiff'.split(' ')


def load_dset(root, **kwargs):
    dset_constructors = {'swiss_roll': SwissRoll, 'mnist': MNIST, 'fashionmnist': FASHIONMNIST, 'cifar10': CIFAR10, 'cifar100': CIFAR100}
    if root.lower() in dset_constructors.keys():
        return dset_constructors[root.lower()](**kwargs)
    else:
        return GenericImageDset(root)


def class_names(**kwargs):
    dset = load_dset(**kwargs)
    classes = dset.classes
    return classes


def center_data(data, norm=True):
    data = data - data.mean()
    if norm:
        data = data / data.std()
    return data


def scale_img(img):
    return img * 2 - 1


def scale_img_inv(img):
    return (img + 1) / 2


def search_imgs(root, sfxs=SFXS):
    pths = []
    for r, _, fs in os.awalk(root):
        for f in fs:
            if any([f.lower().endswith(s) for s in sfxs]):
                pths.append(f)
    return np.sort(pths)


class TVData(Dataset):
    """torchvision dataset adapter"""

    def __init__(self, constructor, *args, cache_dir='./Reflected-Diffusion/cache_datasets', **kwargs):
        super().__init__()
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        self.dset_train = constructor(root=cache_dir, **kwargs)
        self.__N_train = len(self.dset_train)
        self.classes = list(self.dset_train.classes)

    def __len__(self):
        return self.__N_train #+ self.__N_test

    def __getitem__(self, idx):
        x, y = self.dset_train[idx] #if idx < self.__N_train else self.dset_test[idx-self.__N_train]
        x = to_tensor(x)
        #return scale_img(x), y
        return x, y


class MNIST(TVData):

    def __init__(self, *args, cache_dir='cache_datasets', **kwargs):
        super().__init__(datasets.MNIST, *args, cache_dir=cache_dir, download=True, **kwargs)

class FASHIONMNIST(TVData):

    def __init__(self, *args, cache_dir='cache_datasets', **kwargs):
        super().__init__(datasets.FashionMNIST, *args, cache_dir=cache_dir,  download=True, **kwargs)

class CIFAR10(TVData):

    def __init__(self, *args, cache_dir='cache_datasets', **kwargs):
        super().__init__(datasets.CIFAR10, *args, cache_dir=cache_dir, download=True, **kwargs)


class CIFAR100(TVData):

    def __init__(self, *args, cache_dir='cache_datasets', **kwargs):
        super().__init__(datasets.CIFAR100, *args, cache_dir=cache_dir, **kwargs)


class GenericImageDset(Dataset):
    """loads all images in subdirs of root, naming convention: <class_label>_<img_id>.<suffix in SFXS>"""

    def __init__(self, root, *args, suffixes=SFXS, **kwargs):
        super().__init__()
        self.paths = search_imgs(root, sfxs=suffixes)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pth = self.paths[idx]
        y = int(pth.split('_')[0])
        x = default_loader(pth)
        return scale_img(to_tensor(x)), y