import os
import itk
import numpy as np
import json
from matplotlib import cm
from monai.transforms import (
    Compose,
    ScaleIntensityRange,
)
from monai.transforms.transform import Transform
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.nifty_folder import make_dataset_numpy


class LoadNumpyArray(Transform):
    def __init__(self):
        pass

    def __call__(self, path):
        return np.load(path)


def get_paths(list_scans, data_groups, opt):  # data_group_to_exclude,test_group,
    if opt.isTrain is True:
        return [
            str1 for str1 in list_scans if any(str2 in str1 for str2 in data_groups)
        ]  # _to_exclude
    if hasattr(opt, "validation") is True:
        return [
            str1
            for str1 in list_scans
            if any(str2 in str1 for str2 in data_groups[opt.fold])
        ]
    else:
        return [
            str1 for str1 in list_scans if any(str2 in str1 for str2 in data_groups)
        ]  # test_group


def construct_index_list(paths, max_size):
    size = 0
    index_list = []
    for path in paths:
        slices = int(path.split("_")[-1].split(".npy")[0])
        size += slices
        index_list.extend([(path, slice) for slice in range(slices)])
        if size > max_size:
            break
    return index_list, size


class UnalignedNumpyDataset(BaseDataset):
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

        if opt.localisation == "all":
            self.A_paths = []
            self.B_paths = []
            for opt.localisation in [
                "ORL",
                "Pelvis",
                "Crane",
                "Abdomen",
                "Thorax",
                "Sein",
            ]:
                """if opt.localisation == "ORL" or opt.localisation == 'Pelvis':
                    opt.dataroot = '../../../data/processed'
                else:
                    opt.dataroot = '/mnt/other_partition/data/processed/"""
                self.dir_A = os.path.join(
                    opt.dataroot, opt.localisation, "MVCT_npy"  # opt.phase + "A_npy" #
                )  # create a path '/path/to/data/trainA'
                print("Data path", self.dir_A)
                self.dir_B = os.path.join(
                    opt.dataroot,
                    opt.localisation,
                    "KVCT_fitted_npy",  # opt.phase + "B_npy" #
                )  # create a path '/path/to/data/trainB'
                with open(
                    "../datalist/data_%s_%s_%s.json"
                    % (opt.phase, opt.datasplit, opt.localisation),
                    "r",
                ) as fp:
                    data_groups = json.load(fp)
                # data_group_to_exclude =  test_group#+ data_groups[opt.fold]
                list_scans = sorted(make_dataset_numpy(self.dir_A))  # self.A_paths
                path_A_loc = get_paths(
                    list_scans, data_groups, opt
                )  # data_group_to_exclude,test_group,
                self.A_paths.extend(path_A_loc)
                list_scans_b = sorted(make_dataset_numpy(self.dir_B))  # self.B_paths
                path_B_loc = get_paths(
                    list_scans_b, data_groups, opt
                )  # data_group_to_exclude,test_group,
                self.B_paths.extend(path_B_loc)
        else:
            self.dir_A = os.path.join(
                opt.dataroot, opt.localisation, "MVCT_npy"  # opt.phase + "A_npy" #
            )  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(
                opt.dataroot,
                opt.localisation,
                "KVCT_fitted_npy",  # opt.phase + "B_npy" # test outpainting KVCT full FOV
            )  # create a path '/path/to/data/trainB'
            with open(
                "../datalist/data_%s_%s_%s.json"
                % (opt.phase, opt.datasplit, opt.localisation),
                "r",
            ) as fp:
                data_groups = json.load(fp)

            """with open("../data_test_%s_%s.json" % (opt.datasplit, opt.localisation), "r") as fp:
                test_group = json.load(fp)"""

            # data_group_to_exclude =  test_group#+ data_groups[opt.fold]
            list_scans = sorted(make_dataset_numpy(self.dir_A))  # self.A_paths
            self.A_paths = get_paths(
                list_scans, data_groups, opt
            )  # data_group_to_exclude,test_group,
            list_scans_b = sorted(make_dataset_numpy(self.dir_B))  # self.B_paths
            self.B_paths = get_paths(
                list_scans_b, data_groups, opt
            )  # data_group_to_exclude,test_group,
        self.A_index, self.A_size = construct_index_list(
            self.A_paths,
            opt.max_dataset_size,
        )
        self.B_index, self.B_size = construct_index_list(
            self.B_paths,
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
        print(self.A_index.__len__(), self.A_paths.__len__())
        print(self.B_index.__len__(), self.B_paths.__len__())

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
        index_A = index % self.A_size
        A_path = self.A_index[index_A][0]  # make sure index is within then range
        A_slice = self.A_index[index_A][1]
        # index_B = index % self.B_size
        B_path = self.B_index[index_A][0]  # make sure index is within then range
        B_slice = self.B_index[index_A][1]
        transform = Compose(
            [
                LoadNumpyArray(),
                ScaleIntensityRange(
                    a_min=-600.0,
                    a_max=400.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # ToTensor(),
            ]
        )
        breakpoint()
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
