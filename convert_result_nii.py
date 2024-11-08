import glob
import os
from pprint import pprint
import SimpleITK as sitk
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    description="convert generated pseudo kVCT to nifty file."
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="model_name/experiment_name if experiment name",
)
parser.add_argument(
    "--localization",
    type=str,
    required=True,
    help="anatomical localization of testing data",
)
args = parser.parse_args()


def insensitive_glob(pattern):
    def either(c):
        return "[%s%s]" % (c.lower(), c.upper()) if c.isalpha() else c

    return glob.glob("".join(map(either, pattern)))


def scaleIntensityRange(img, a_min: float, a_max: float, b_min: float, b_max: float):
    """
    Apply specific intensity scaling to the whole numpy array.
    Scaling from [a_min, a_max] to [b_min, b_max].

    Args:
        img:   image to apply scaling
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
    """

    img = (img - a_min) / (a_max - a_min)
    img = img * (b_max - b_min) + b_min
    return img


model_name = args.model_name
localization = args.localization
"ORL"
globResult = sorted(
    insensitive_glob(
        "{}/*_fake_B_outpainted.png".format(model_name)
    )  # results/{}/test_latest/images/*_fake_B.png
)
previousfile = ""
for file in globResult:
    patient = os.path.split(file)[1].split("-")[0]
    if previousfile == "":
        patient_list = set()
        patient_list.add(patient)
        previousfile = patient
    if patient != previousfile:  ##New patient
        patient_list.add(patient)
        previousfile = patient
print(patient_list)

for patient in patient_list:
    globPatient = sorted(
        insensitive_glob("{}/{}*_fake_B_outpainted.png".format(model_name, patient))
    )
    kvct_path = (
        "../../../data/processed/{}/KVCT_fitted/{}-fitted_mask_kvct.nii.gz".format(
            localization, patient
        )
    )
    kv_source_image = sitk.ReadImage(kvct_path)  # kVCT for spatial information of image
    nb_slices = kv_source_image.GetSize()
    volume_patient = np.empty([nb_slices[2], nb_slices[0], nb_slices[1]])
    for files in globPatient:
        print(files)
        slice_nb = int(os.path.split(files)[1].split(".")[0].split("_")[2])
        imgi = Image.open(files)
        slice = np.asarray(imgi)
        # breakpoint()
        slice = scaleIntensityRange(
            slice, a_min=0, a_max=255, b_min=-160, b_max=240
        )  # -600, b_max=400 cyclegan
        # print(slice.shape, slice[0, 0, :])
        print("Slice", slice.min(), slice.max(), slice.shape)
        volume_patient[slice_nb, :, :] = slice[:, :]
        print("Volume", volume_patient.min(), volume_patient.max())

    img = sitk.GetImageFromArray(volume_patient)
    img.CopyInformation(srcImage=kv_source_image)
    # print(img.GetSpacing())
    saveresult = "{}/{}_fake_B_outpainted.nii.gz".format(model_name, patient)
    sitk.WriteImage(img, saveresult)
"""
    
    
    if previousfile == "": ##First file
        kvct_path = "../../../data/processed/{}/KVCT_fitted/{}-fitted_mask_kvct.nii.gz".format(localisation, previousfile)
        kv_source_image = sitk.ReadImage(kvct_path)  # kVCT for spatial information of image
        nb_slices = kv_source_image.GetSize()
        volume_patient = np.empty([nb_slices[2], nb_slices[0], nb_slices[1]])
        pr

    if patient == previousfile:
        print(patient, slice_nb)
        #volume_patient["patient"] = patient
        #print(volume_patient.keys())
        volume_patient[slice_nb,:,:] = slice[:,:,0]"""
"""plt.figure(1)
        plt.imshow(slice[:,:,0],cmap="gray", interpolation="none")
        plt.show()"""

"""print(volume_patient.shape)
        previousfile = patient
    if patient != previousfile: ##New patient
             
        img = sitk.GetImageFromArray(volume_patient)
        img.CopyInformation(srcImage=kv_source_image)
        print(img.GetSpacing())
        saveresult = "results/{}/{}/test_latest/images/{}_fake_B.nii.gz".format(
            model_name, experiment_name, previousfile
        )
        sitk.WriteImage(img,saveresult)
        #write_image_pseudo_kvct(saveresult, previousfile, img)
        print(dir(kv_source_image))

print(volume_patient) """
