import os
import random

import itk
import numpy as np
from matplotlib import cm
from monai.transforms import (
    Compose,
    Resize,
    ScaleIntensityRange,
    ShiftIntensity,
    ToTensor,
)
from monai.transforms.transform import Transform
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.nifty_folder import make_dataset


class ITKImageToNumpyd(Transform):
    def __init__(self):
        pass

    def __call__(self, data):
        return itk.array_from_image(data)


class LoadITKImage(Transform):
    def __init__(self, pixel_type=itk.F):
        self.pixel_type = pixel_type

    def __call__(self, key):
        d = itk.imread(key, pixel_type=self.pixel_type)
        return d


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.pixel_type = itk.F
        self.dir_A = os.path.join(
            opt.dataroot, opt.phase + "A"  # "MVCT" #
        )  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(
            opt.dataroot, opt.phase + "B"  # "KVCT_fitted" #
        )  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(
            make_dataset(self.dir_A)
        )  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(
            make_dataset(self.dir_B)
        )  # load images from '/path/to/data/trainB'
        self.A_size = 0
        self.A_index = []
        for path in self.A_paths:
            image = itk.array_from_image(itk.imread(path, pixel_type=self.pixel_type))
            self.A_size += image.shape[0]  # get the size of dataset A
            for slice in range(image.shape[0]):
                self.A_index.append((path, slice))
            if self.A_size > opt.max_dataset_size:
                break
        self.B_size = 0
        self.B_index = []
        for path in self.B_paths:
            image = itk.array_from_image(itk.imread(path, pixel_type=self.pixel_type))
            self.B_size += image.shape[0]  # get the size of dataset B
            for slice in range(image.shape[0]):
                self.B_index.append((path, slice))
            if self.B_size > opt.max_dataset_size:
                break

        btoA = self.opt.direction == "BtoA"
        input_nc = (
            self.opt.output_nc if btoA else self.opt.input_nc
        )  # get the number of channels of input image
        output_nc = (
            self.opt.input_nc if btoA else self.opt.output_nc
        )  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        print(self.A_size, self.B_size)
        print(self.A_index.__len__())

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # print(index)
        # print("result:",index % self.A_size)
        A_path = self.A_index[index % self.A_size][
            0
        ]  # make sure index is within then range
        A_slice = self.A_index[index % self.A_size][1]
        index_B = index % self.B_size
        B_path = self.B_index[index % self.A_size][
            0
        ]  # make sure index is within then range
        B_slice = self.B_index[index % self.A_size][1]
        # print(A_path, A_slice, B_path, B_slice)
        # print(B_path, B_slice)
        transform = Compose(
            [
                LoadITKImage(),
                ITKImageToNumpyd(),
                ScaleIntensityRange(
                    a_min=-600.0,
                    a_max=400.0,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                ),
                # ToTensor(),
            ]
        )
        A_img = transform(A_path)[A_slice, :, :]
        B_img = transform(B_path)[B_slice, :, :]
        # print(A_img.size())
        # print(B_img.size())
        # print("Min before image",A_img.min())
        # print("Max before image",A_img.max())
        im = Image.fromarray(np.uint8(cm.gist_earth(A_img) * 255))
        imb = Image.fromarray(np.uint8(cm.gist_earth(B_img) * 255))
        # print("Before image:",A_img.min(), A_img.max())#print("Min after image",)
        # print(im.__sizeof__())
        # print(imb.__sizeof__())
        # print("After  image:",im.getextrema())
        A_path_split = os.path.splitext(A_path)
        A_path_split2 = os.path.splitext(A_path_split[0])
        A_path_slice = os.path.join(
            A_path_split2[0] + "_" + str(A_slice) + A_path_split2[1] + A_path_split[1]
        )
        # print(A_path_slice)
        B_path_split = os.path.splitext(B_path)
        B_path_split2 = os.path.splitext(B_path_split[0])
        B_path_slice = os.path.join(
            B_path_split2[0] + "_" + str(B_slice) + B_path_split2[1] + B_path_split[1]
        )
        # print(B_path_slice)
        return {
            "A": self.transform_A(im),
            "B": self.transform_B(imb),
            "A_paths": A_path_slice,
            "B_paths": B_path_slice,
        }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
