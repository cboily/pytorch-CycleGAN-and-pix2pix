import os
import random
import time
import itk
import json
import numpy as np
from matplotlib import cm
from monai.transforms import (
    Compose,
    ScaleIntensityRange,
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


def deterministic_split(files_list, k, seed_value=0):
    random.seed(seed_value)

    shuffled_files = files_list[:]
    random.shuffle(shuffled_files)
    random.seed(None)
    n = len(shuffled_files)
    step = n // k

    sublists = [shuffled_files[i : i + step] for i in range(0, n, step)]

    if len(sublists) > k:
        sublists[k - 1].extend(sublists[k:])
        sublists = sublists[:k]

    return sublists


def get_paths(list_scans, data_group_to_exclude, data_groups, opt):
    if opt.isTrain is True:
        return [
            str1
            for str1 in list_scans
            if not any(str2 in str1 for str2 in data_group_to_exclude)
        ]
    else:
        if opt.validation is True:
            return [
                str1
                for str1 in list_scans
                if any(str2 in str1 for str2 in data_groups[opt.fold])
            ]
        else:
            return [
                str1 for str1 in list_scans if any(str2 in str1 for str2 in data_groups)
            ]


def construct_index_list(paths, pixel_type, localisation, ct_type, max_size):
    with open("../index_%s_%s.json" % (localisation, ct_type), "r") as f:
        image_data_list = json.load(f)

    image_data_dict = {item["path"]: item["size"] for item in image_data_list}
    size = 0
    index_list = []
    # image_data_list=[]
    for path in paths:
        if path in image_data_dict:
            image_shape = image_data_dict[path]
            size += image_shape
        # image = itk.array_from_image(itk.imread(path, pixel_type=pixel_type))

        index_list.extend([(path, slice) for slice in range(image_shape)])
        # image_data_list.append({"path": path, "size": image.shape[0]})
        if size > max_size:
            break
    """localisation = "ORL"
    with open("index_%s_%s.json" % (localisation, ct_type ), "w") as json_file:
            json.dump(image_data_list, json_file, indent=4)"""
    return index_list, size


class UnalignedKFoldDataset(BaseDataset):
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
            opt.dataroot, "MVCT"  # opt.phase + "A"  #
        )  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(
            opt.dataroot, "KVCT_fitted"  # opt.phase + "B"  #
        )  # create a path '/path/to/data/trainB'
        with open("../data_%s_%s.json" % (opt.phase, opt.localisation), "r") as fp:
            data_groups = json.load(fp)

        data_group_to_exclude = data_groups[opt.fold]
        list_scans = sorted(make_dataset(self.dir_A))
        self.A_paths = get_paths(list_scans, data_group_to_exclude, data_groups, opt)

        list_scans_b = sorted(make_dataset(self.dir_B))
        self.B_paths = get_paths(list_scans_b, data_group_to_exclude, data_groups, opt)

        self.A_index, self.A_size = construct_index_list(
            self.A_paths,
            self.pixel_type,
            opt.localisation,
            "MVCT",
            opt.max_dataset_size,
        )
        self.B_index, self.B_size = construct_index_list(
            self.B_paths,
            self.pixel_type,
            opt.localisation,
            "KVCT_fitted",
            opt.max_dataset_size,
        )
        btoA = self.opt.direction == "BtoA"
        input_nc = (
            self.opt.output_nc if btoA else self.opt.input_nc
        )  # get the number of channels of input image
        output_nc = (
            self.opt.input_nc if btoA else self.opt.output_nc
        )  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

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
        if self.B_size < self.A_size:
            index_range = index % self.B_size
        else:
            index_range = index % self.A_size
        A_path = self.A_index[index_range][0]  # make sure index is within then range
        A_slice = self.A_index[index_range][1]
        B_path = self.B_index[index_range][0]  # make sure index is within then range
        B_slice = self.B_index[index_range][1]
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
            ]
        )
        A_img = transform(A_path)[A_slice, :, :]
        B_img = transform(B_path)[B_slice, :, :]
        im = Image.fromarray(np.uint8(cm.gist_earth(A_img) * 255))
        imb = Image.fromarray(np.uint8(cm.gist_earth(B_img) * 255))
        A_path_split = os.path.splitext(A_path)
        A_path_split2 = os.path.splitext(A_path_split[0])
        A_path_slice = os.path.join(
            A_path_split2[0] + "_" + str(A_slice) + A_path_split2[1] + A_path_split[1]
        )
        B_path_split = os.path.splitext(B_path)
        B_path_split2 = os.path.splitext(B_path_split[0])
        B_path_slice = os.path.join(
            B_path_split2[0] + "_" + str(B_slice) + B_path_split2[1] + B_path_split[1]
        )

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
