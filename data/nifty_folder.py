"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
from monai.transforms import LoadImage
import os

IMG_EXTENSIONS = [
    ".nii.gz",
    ".nii",
]

NUMPY_EXTENSIONS = [".npy"]


def is_nifty_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in NUMPY_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_nifty_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
            if is_numpy_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images[: min(max_dataset_size, len(images))]


def make_dataset_numpy(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    fnames = os.scandir(dir)
    for fname in fnames:
        if is_numpy_file(fname.name):
            images.append(fname.path)

    return images[: min(max_dataset_size, len(images))]


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_nifty_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images[: min(max_dataset_size, len(images))]


def default_loader(path):
    return LoadImage(image_only=True)(path)


class NiftyFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 nifty in: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
