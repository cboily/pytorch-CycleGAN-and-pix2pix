import json
import os

import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.nifty_folder import make_dataset_numpy
from PIL import Image
from matplotlib import cm
from monai.transforms.compose import Compose
from monai.transforms.intensity.array import ScaleIntensityRange
from data.unaligned_numpy_dataset import LoadNumpyArray, get_paths, construct_index_list


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
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
        else:
            self.dir_A = os.path.join(
                opt.dataroot, opt.localisation  # , "MVCT_npy"  # opt.phase + "A_npy" #
            )  # create a path '/path/to/data/trainA'
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
        self.A_index, self.A_size = construct_index_list(
            self.A_paths,
            opt.max_dataset_size,
        )
        input_nc = (
            self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        )
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        index_A = index % self.A_size
        A_path = self.A_index[index_A][0]  # make sure index is within then range
        A_slice = self.A_index[index_A][1]
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
        A_img = transform(A_path)[A_slice, :, :]
        im = Image.fromarray(np.uint8(cm.gist_earth(A_img) * 255))
        A_path_split = os.path.splitext(A_path)
        A_path_slice = os.path.join(
            A_path_split[0] + "_" + str(A_slice) + A_path_split[1]
        )
        print("Path", A_path_slice)
        return {"A": self.transform(im), "A_paths": A_path_slice}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size
