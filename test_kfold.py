"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from pprint import pprint
import SimpleITK as sitk
import numpy as np
from typing import List, Dict, Tuple
import json
import matplotlib.pyplot as plt

# from torchmetrics import MeanAbsoluteError
import torch
from monai.transforms import Compose, ScaleIntensityRange
from torchmetrics import (
    MeanSquaredError,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.functional import mean_absolute_error

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images

try:
    import wandb
except ImportError:
    print(
        'Warning: wandb package cannot be found. The option "--use_wandb" will result in error.'
    )


def euler_3d_registration(fixed_bone, moving_bone, moving_body):
    fixed = sitk.GetImageFromArray(fixed_bone)  # realB_image
    moving = sitk.GetImageFromArray(moving_bone)  # fakeB_image
    moving_full = sitk.GetImageFromArray(moving_body)

    # Create an image registration method
    R = sitk.ImageRegistrationMethod()

    # Set the metric as correlation
    R.SetMetricAsCorrelation()

    # Set optimizer parameters
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromIndexShift()

    # Initialize the transformation
    tx = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)

    # Execute the registration
    outTx = R.Execute(fixed, moving)

    # Output registration information
    """print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f"Iteration: {R.GetOptimizerIteration()}")
    print(f"Metric value: {R.GetMetricValue()}")"""

    # Write the transformation to a file
    # sitk.WriteTransform(outTx, "transform.txt")

    # Resampling
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-601)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    out_full = resampler.Execute(moving_full)

    return out, out_full


def calculate_metrics(fakeB, realB):
    result_data = {}
    body_mask = realB > -600
    background_mask = ~body_mask
    # back_mask_f = fakeB > -600
    realB_back = realB[background_mask]
    fakeB_back = fakeB[background_mask]
    realB = realB * body_mask
    fakeB = fakeB * body_mask
    # print("Fa", fakeB.shape, fakeB.min(), fakeB.max())  # fakeB,
    threshold_value = 150
    bone_mask = realB >= threshold_value
    tissu_mask_s = (
        realB < threshold_value
    )  # .cpu() &realB.cpu() >-600).to(device="cuda:0")#
    tissu_mask = tissu_mask_s & body_mask
    body_fraction = np.mean(body_mask == 1)
    back_fraction = np.mean(background_mask == 1)
    tissu_fraction = np.mean(tissu_mask == 1)
    bone_fraction = np.mean(bone_mask == 1)
    """print(
        "background",
        back_fraction,
        "body",
        body_fraction,
        "bone",
        bone_fraction,
        "tissu",
        tissu_fraction,
    )"""
    if bone_fraction != 0.0:
        # bone_mask_fake = (fakeB >= threshold_value)
        # bone_mask = bone_mask_fake | bone_mask_real
        fakeB_tissu = fakeB[tissu_mask]  # *
        fakeB_bone = fakeB[bone_mask]  # *
        realB_tissu = realB[tissu_mask]  # *
        realB_bone = realB[bone_mask]  # *
        """print(
            "real_tissu",
            realB_tissu.min(),
            realB_tissu.max(),
            realB_tissu.size(),
        )
        print(
            "fake_tissu",
            fakeB_tissu.min(),
            fakeB_tissu.max(),
            fakeB_tissu.size(),
            np.mean(fakeB_tissu == 0),
        )

        print(
            "bone_real",
            realB_bone.min(),
            realB_bone.max(),
            bone_mask.size(),
            realB_bone.size(),
        )
        print(
            "bone_fake", fakeB_bone.min(), fakeB_bone.max(), fakeB_bone.size()
        )
        print(
            "background", fakeB_back.min(), fakeB_back.max(), fakeB_back.size()
        )
        import matplotlib.pyplot as plt

        plt.imshow(fakeB[0, 0, :, :].cpu(), cmap='gray')
        plt.show()
        plt.imshow(fakeB_back[0, 0, :, :].cpu(), cmap="gray")
        plt.show()
        plt.imshow(tissu_mask[0, 0, :, :].cpu())  # realB_tissu
        plt.show()
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(fakeB_tissu[0, 0, :, :].cpu(), cmap="gray")
        plt.title("Image tissu")
        plt.subplot(2, 2, 2)
        plt.imshow(realB_tissu[0, 0, :, :].cpu(), cmap="gray")
        plt.subplot(2, 2, 3)
        plt.imshow(fakeB_bone[0, 0, :, :].cpu(), cmap="gray")
        plt.title("Image Bone")
        plt.subplot(2, 2, 4)
        plt.imshow(realB_bone[0, 0, :, :].cpu(), cmap="gray")
        plt.show()"""

        result_data["MAE"] = mean_absolute_error(fakeB, realB).item()
        testssim = StructuralSimilarityIndexMeasure(
            reduction="none", return_full_image=True
        ).to(device="cuda:0")
        testpsnr = PeakSignalNoiseRatio(reduction="none", dim=0, data_range=1000).to(
            device="cuda:0"
        )
        testrmse = MeanSquaredError(squared=False).to(device="cuda:0")
        result_data["RMSE"] = testrmse(fakeB, realB).item()
        ssim_b, ssim = testssim(fakeB, realB)
        result_data["SSIM"] = ssim_b.item()
        psnr = testpsnr(fakeB, realB)
        result_data["PSNR"] = torch.mean(psnr[torch.flatten(body_mask)]).item()  #
        # print(result_data["PSNR"].size(), torch.mean(result_data["PSNR"]).item() )
        result_data["MAE_back"] = mean_absolute_error(fakeB_back, realB_back).item()
        result_data["MAE_bone"] = mean_absolute_error(fakeB_bone, realB_bone).item()
        result_data["MAE_tissu"] = mean_absolute_error(fakeB_tissu, realB_tissu).item()
        result_data["RMSE_bone"] = testrmse(fakeB_bone, realB_bone).item()
        result_data["RMSE_tissu"] = testrmse(fakeB_tissu, realB_tissu).item()
        result_data["RMSE_back"] = testrmse(fakeB_back, realB_back).item()
        result_data["SSIM_bone"] = torch.mean(
            ssim[bone_mask]
        ).item()  # testssim(fakeB_bone, realB_bone).item()
        result_data["SSIM_tissu"] = torch.mean(
            ssim[tissu_mask]
        ).item()  # testssim(fakeB_tissu, realB_tissu).item()
        result_data["SSIM_back"] = torch.mean(ssim[background_mask]).item()
        # result_data["PSNR_bone"] = testpsnr(fakeB_bone, realB_bone).item()
        # result_data["PSNR_tissu"] = testpsnr(fakeB_tissu, realB_tissu).item()
        result_data["PSNR_bone"] = torch.mean(
            psnr[torch.flatten(bone_mask)]
        ).item()  # testpsnr(fakeB_bone, realB_bone).item()
        result_data["PSNR_tissu"] = torch.mean(psnr[torch.flatten(tissu_mask)]).item()
        result_data["background"] = back_fraction
        result_data["tissu"] = tissu_fraction
        result_data["bone"] = bone_fraction
        # print('tissu',result_data["SSIM_tissu"].shape, result_data["SSIM_tissu"],'background', result_data["SSIM_back"].shape, result_data["SSIM_back"])
        # print(result_data)
    return result_data


def calculate_mean_metrics(data_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculates the mean of each metric from a list of dictionaries."""

    results = []

    for sublist in data_list:
        metric_values = {
            metric: []
            for metric in [
                "MAE",
                "MAE_bone",
                "MAE_tissu",
                "MAE_back",
                "PSNR",
                "PSNR_bone",
                "PSNR_tissu",
                "RMSE",
                "RMSE_back",
                "RMSE_bone",
                "RMSE_tissu",
                "SSIM",
                "SSIM_bone",
                "SSIM_tissu",
                "SSIM_back",
            ]
        }

        for test_image in sublist:
            for metrics in test_image.values():
                for metric in metric_values:
                    value = metrics.get(metric, 0)
                    if not np.isinf(value):
                        metric_values[metric].append(value)

        mean_values = {
            metric: np.mean(values) for metric, values in metric_values.items()
        }
        stddev_values = {
            metric: np.std(values) for metric, values in metric_values.items()
        }

        sublist_result = {
            f"mean_{metric}": mean_values[metric] for metric in mean_values
        }
        sublist_result.update(
            {f"stddev_{metric}": stddev_values[metric] for metric in stddev_values}
        )

        results.append(sublist_result)

    return results


def rank_data_by_metrics(
    data_list: List[Dict[str, float]]
) -> List[Tuple[Dict[str, float], int]]:
    """Ranks dictionaries in the list based on defined metrics and returns them along with their original indices."""

    def score(item: Tuple[Dict[str, float], int]) -> float:
        """Compute a score for the dictionary based on its metrics."""
        d = item[0]
        MAE_score = 1e6 - d["mean_MAE"]  # Lower MAE is better
        PSNR_score = d["mean_PSNR"]  # Higher PSNR is better
        RMSE_score = 1e6 - d["mean_RMSE"]  # Lower RMSE is better
        SSIM_score = d["mean_SSIM"] * 1e6  # SSIM closer to 1 is better
        return MAE_score + PSNR_score + RMSE_score + SSIM_score

    # Pair each dictionary with its original index
    indexed_data = [(d, idx) for idx, d in enumerate(data_list)]

    # Sort list by score in descending order
    return sorted(indexed_data, key=score, reverse=True)


def plot_metrics(ranked_data: List[Tuple[Dict[str, float], int]]):
    """Plots metrics for each dictionary in the list using bar plots in a single figure."""

    indices = [item[1] for item in ranked_data]

    metrics = ["mean_MAE", "mean_PSNR", "mean_RMSE", "mean_SSIM"]
    titles = ["Mean MAE", "Mean PSNR", "Mean RMSE", "Mean SSIM"]
    import matplotlib.pyplot as plt

    # Set up a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # Iterate through each metric and create its respective subplot
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        values = [item[0][metric] for item in ranked_data]
        axes[i].bar(indices, values, color="skyblue", edgecolor="black")
        axes[i].set_title(title)
        axes[i].set_xlabel("Original Index")
        axes[i].set_ylabel("Metric Value")
        axes[i].grid(axis="y")

    plt.tight_layout()
    plt.show()


def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} " + f"= {method.GetMetricValue():10.5f}")


def iteration_callback(filter):
    print("\r{0:.2f}".format(filter.GetMetricValue()), end="")


with torch.no_grad():
    if __name__ == "__main__":
        opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        opt.num_threads = 0  # test code only supports num_threads = 0
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = (
            True  # no flip; comment this line if results on flipped images are needed.
        )
        opt.display_id = (
            -1
        )  # no visdom display; the test code saves the results to a HTML file.

        opt.num_folds = 1
        opt.seed = 53493403
        opt.isTrain = False
        result = []
        result_mv = []
        result_ana = []
        result_ana_mv = []
        result_reg = []
        # name = opt.name
        for k in range(0, opt.num_folds):
            result.append([])
            result_mv.append([])
            result_ana.append([])
            result_ana_mv.append([])
            result_reg.append([])
            opt.fold = k
            print(f"FOLD {opt.fold}")
            print("--------------------------------")
            # opt.name = name + str(opt.fold)
            print("Name:", opt.name)
            dataset = create_dataset(
                opt
            )  # create a dataset given opt.dataset_mode and other options
            # opt.serial_batches = False
            # print("Size Test subset", opt.num_test)
            model = create_model(
                opt
            )  # create a model given opt.model and other options
            model.setup(
                opt
            )  # regular setup: load and print networks; create schedulers

            # initialize logger
            if opt.use_wandb:
                wandb_run = (
                    wandb.init(
                        project=opt.wandb_project_name, name=opt.name, config=opt
                    )
                    if not wandb.run
                    else wandb.run
                )
                wandb_run._label(repo="CycleGAN-and-pix2pix")

            # create a website
            web_dir = os.path.join(
                opt.results_dir,
                opt.name,
                "{}_fold_{}_{}_{}".format(
                    opt.phase, opt.fold, opt.epoch, opt.localisation
                ),
            )  # define the website directory
            if opt.load_iter > 0:  # load_iter is 0 by default
                web_dir = "{:s}_iter{:d}".format(web_dir, opt.load_iter)
            print("creating web directory", web_dir)
            webpage = html.HTML(
                web_dir,
                "Experiment = %s, Phase = %s, Fold = %s, Epoch = %s"
                % (opt.name, opt.phase, opt.fold, opt.epoch),
            )

            # test with eval mode. This only affects layers like batchnorm and dropout.
            # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
            # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
            if opt.eval:
                model.eval()
            result_fold = {}
            result_fold_mv = {}
            result_fold_ana = {}
            result_fold_ana_mv = {}
            result_fold_reg = {}
            prev_patient_name = None
            stacked_fakeBs = []
            stacked_realBs = []
            stacked_fakeBs_bone = []
            stacked_realBs_bone = []
            stacked_name_file = []
            for i, data in enumerate(dataset):
                if i >= opt.num_test:  # only apply our model to opt.num_test images.
                    break
                model.set_input(data)  # unpack data from data loader
                data_name = os.path.split(str(data["A_paths"][0]))[1]
                patient_id = data_name.split("-")[0]
                # print('Patient', patient_id, prev_patient_name)
                # print(data_name)
                # result_data
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results

                out_transforms = Compose(
                    [
                        ScaleIntensityRange(
                            a_min=-1.0,
                            a_max=1.0,
                            b_min=-600.0,
                            b_max=400.0,
                            # clip=True,
                        ),
                    ]
                )
                fakeB = out_transforms(visuals["fake_B"])
                realB = out_transforms(visuals["real_B"])
                realA = out_transforms(visuals["real_A"])
                fakeA = out_transforms(visuals["fake_A"])
                recB = out_transforms(visuals["rec_B"])
                threshold_value = 150
                bone_mask = realB >= threshold_value
                fakeB_bone = fakeB * bone_mask
                realB_bone = realB * bone_mask
                if prev_patient_name is None:
                    prev_patient_name = patient_id
                if patient_id != prev_patient_name:
                    # Reset the stacked_fakeBs list when data_name changes
                    if len(stacked_fakeBs) >= 4:
                        print("Calculate patient and reset")
                        sliding_stack_fakeB = np.stack(stacked_fakeBs, axis=2)
                        sliding_stack_realB = torch.stack(stacked_realBs, axis=2)
                        sliding_stack_fakeB_bone = np.stack(stacked_fakeBs_bone, axis=2)
                        sliding_stack_realB_bone = np.stack(stacked_realBs_bone, axis=2)
                        """print(
                            "Sliding stack, fakeB",
                            sliding_stack_fakeB.shape,
                            sliding_stack_fakeB.dtype,
                            sliding_stack_fakeB_bone.shape,
                            "realB",
                            sliding_stack_realB.shape,
                            sliding_stack_realB_bone.shape,
                        )"""
                        out, out_full = euler_3d_registration(
                            sliding_stack_realB_bone[0, 0, :, :, :],
                            sliding_stack_fakeB_bone[0, 0, :, :, :],
                            sliding_stack_fakeB[0, 0, :, :, :],
                        )
                        aligned_fakeB_array_full = sitk.GetArrayFromImage(out_full)
                        aligned_fakeB_array_full_t = fakeB.new_tensor(
                            [[np.array(aligned_fakeB_array_full)]]
                        )
                        for slice in range(0, aligned_fakeB_array_full.shape[0]):
                            zero_fraction = np.mean(
                                aligned_fakeB_array_full[slice, :, :] == -601
                            )
                            if zero_fraction <= 0.05:
                                metrics_reg = calculate_metrics(
                                    aligned_fakeB_array_full_t[:, :, slice, :, :],  #
                                    sliding_stack_realB[
                                        :, :, slice, :, :
                                    ],  # aligned_realB_array_full_t[:, :, slice, :, :],  #
                                )
                                if result_reg:
                                    result_fold_reg[
                                        stacked_name_file[slice]
                                    ] = metrics_reg
                            else:
                                print(
                                    "Error fraction:",
                                    stacked_name_file[slice],
                                    zero_fraction,
                                )
                    stacked_fakeBs = []
                    stacked_realBs = []
                    stacked_fakeBs_bone = []
                    stacked_realBs_bone = []
                    stacked_name_file = []
                    prev_patient_name = patient_id

                # Append transformed fakeB image to the list
                stacked_fakeBs.append(fakeB)
                stacked_realBs.append(realB)
                stacked_fakeBs_bone.append(fakeB_bone)
                stacked_realBs_bone.append(realB_bone)
                stacked_name_file.append(data_name)
                # print(prev_patient_name, len(stacked_fakeBs), fakeB.shape, fakeB.dtype)
                # Ensure the stack doesn't exceed a length of 7
                if len(stacked_fakeBs) > 7:
                    stacked_fakeBs.pop(0)
                    stacked_realBs.pop(0)
                    stacked_fakeBs_bone.pop(0)
                    stacked_realBs_bone.pop(0)
                    stacked_name_file.pop(0)

                # Create the sliding stack of fakeB images
                if len(stacked_fakeBs) == 7:
                    sliding_stack_fakeB = np.stack(stacked_fakeBs, axis=2)
                    sliding_stack_realB = np.stack(stacked_realBs, axis=2)
                    sliding_stack_realB_t = torch.stack(stacked_realBs, axis=2)
                    sliding_stack_fakeB_bone = np.stack(stacked_fakeBs_bone, axis=2)
                    sliding_stack_realB_bone = np.stack(stacked_realBs_bone, axis=2)
                    """print(
                        "Sliding stack, fakeB",
                        sliding_stack_fakeB.shape,
                        sliding_stack_fakeB.dtype,
                        sliding_stack_fakeB_bone.shape,
                        "realB",
                        sliding_stack_realB.shape,
                        sliding_stack_realB.dtype,
                        sliding_stack_realB_bone.shape,
                    )"""
                    """

                    # Visualize the original realB and aligned fakeB images using matplotlib
                    plt.figure(figsize=(10, 5))

                    plt.subplot(1, 2, 1)
                    plt.imshow(
                        sliding_stack_realB[0, 0, 0, :, :], cmap="gray"
                    )  # Assuming it's a 3D image stack, selecting the first slice
                    plt.title("Original RealB")
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(
                        sliding_stack_fakeB_bone[0, 0, 0, :, :], cmap="gray"
                    )  # Assuming it's a 3D image stack, selecting the first slice
                    plt.title(" FakeB")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.show()"""

                    fixed_full = sitk.GetImageFromArray(
                        sliding_stack_realB[0, 0, :, :, :]
                    )

                    # fixed_image.SetOrigin([0,0,0])
                    # fixed_image.SetDimension

                    # print('fakeB image', fakeB_image.GetSize(), 'realB', realB_image.GetSize())

                    """elastixImageFilter = sitk.ElastixImageFilter()
                    elastixImageFilter.SetFixedImage(realB_image)
                    elastixImageFilter.SetMovingImage(fakeB_image)

                    parameterMapVector = sitk.VectorOfParameterMap()
                    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
                    #parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
                    elastixImageFilter.SetParameterMap(parameterMapVector)

                    elastixImageFilter.Execute()
                    resultImage= elastixImageFilter.GetResultImage()"""

                    """registration_method = sitk.ImageRegistrationMethod()
                    print('Fixed image',fixed_image, fixed_image.GetSize(), fixed_image.GetSpacing(), fixed_image.GetOrigin())
                    transformDomainMeshSize = [8] * moving_image.GetDimension()
                    print('transform', transformDomainMeshSize)
                    # Determine the number of BSpline control points using the physical 
                    # spacing we want for the finest resolution control grid. 
                    grid_physical_spacing = [50.0, 50.0, 50.0] # A control point every 50mm
                    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
                    mesh_size = [int(image_size/grid_spacing + 0.5) \
                                for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
                    
                    # The starting mesh size will be 1/4 of the original, it will be refined by 
                    # the multi-resolution framework.
                    mesh_size = [int(sz/4 + 0.5) for sz in mesh_size]
                    print('Mesh', image_physical_size, mesh_size)
                    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 
                                                                        transformDomainMeshSize = mesh_size, order=3)    
                    print(initial_transform)
                    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
                    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with 
                    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
                    # in the full resolution image we use a mesh that is four times the original size.
                    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                                    inPlace=False,
                                                                    scaleFactors=[1,2,1])

                    registration_method.SetMetricAsMeanSquares()
                    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
                    registration_method.SetMetricSamplingPercentage(0.01)
                    #registration_method.SetMetricFixedMask(body)
                        
                    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
                    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
                    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

                    registration_method.SetInterpolator(sitk.sitkLinear)
                    registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-2, numberOfIterations=100, deltaConvergenceTolerance=0.01)

                    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(registration_method))

                    final_transformation = registration_method.Execute(fixed_image, moving_image)
                    print('\nOptimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
                    
                    transformed_segmentation = sitk.Resample(
                                         fixed_image,
                                         final_transformation, 
                                         sitk.sitkNearestNeighbor,
                                         0.0                                        )"""
                    ##B Spline
                    """transformDomainMeshSize = [8] * moving.GetDimension()
                    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

                    print("Initial Parameters:")
                    print(tx.GetParameters())

                    R = sitk.ImageRegistrationMethod()
                    R.SetMetricAsCorrelation()

                    R.SetOptimizerAsLBFGSB(
                        gradientConvergenceTolerance=1e-5,
                        numberOfIterations=100,
                        maximumNumberOfCorrections=5,
                        maximumNumberOfFunctionEvaluations=1000,
                        costFunctionConvergenceFactor=1e7,
                    )
                    R.SetInitialTransform(tx, True)
                    R.SetInterpolator(sitk.sitkLinear)

                    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

                    outTx = R.Execute(fixed, moving)

                    print("-------")
                    print(outTx)
                    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
                    print(f" Iteration: {R.GetOptimizerIteration()}")
                    print(f" Metric value: {R.GetMetricValue()}")

                    sitk.WriteTransform(outTx, 'transform.txt')

                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(fixed)
                    resampler.SetInterpolator(sitk.sitkLinear)
                    resampler.SetDefaultPixelValue(100)
                    resampler.SetTransform(outTx)

                    out = resampler.Execute(moving)
                    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
                    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
                    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)"""

                    # Define the registration method (rigid registration)
                    """registration_method = sitk.ImageRegistrationMethod()

                    # Define the metric, optimizer, interpolator, and transform for the registration
                    registration_method.SetMetricAsMeanSquares()
                    registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=100)
                    registration_method.SetInterpolator(sitk.sitkLinear)

                    initial_transform = sitk.CenteredTransformInitializer(
                        realB_image,fakeB_image, sitk.Euler3DTransform()
                    )

                    registration_method.SetInitialTransform(initial_transform, inPlace=False)
                    #registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
                    #registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])

                    # Execute the registration
                    final_transform = registration_method.Execute(realB_image, fakeB_image)

                    # Apply the final transform to align the fakeB image onto the realB image
                    registered_fakeB = sitk.Resample(fakeB_image, realB_image, final_transform, sitk.sitkLinear, 0.0)

                    # Get the aligned fakeB array"""
                    out, out_full = euler_3d_registration(
                        sliding_stack_realB_bone[0, 0, :, :, :],
                        sliding_stack_fakeB_bone[0, 0, :, :, :],
                        sliding_stack_fakeB[0, 0, :, :, :],
                    )
                    """fixed = sitk.GetImageFromArray(
                        sliding_stack_realB_bone[0, 0, :, :, :]
                    )  # realB_image
                    # cimg = sitk.Compose(out_full_body, out_full, out_full_body // 2.0 + out_full // 2.0)
                    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
                    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
                    simg3 = sitk.Cast(sitk.RescaleIntensity(fixed_full), sitk.sitkUInt8)
                    simg4 = sitk.Cast(sitk.RescaleIntensity(out_full), sitk.sitkUInt8)
                    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
                    # cimg = sitk.Compose(fixed, out, fixed // 2.0 + out // 2.0)
                    # cimg2 = sitk.Compose(fixed_full, out_full, fixed_full // 2.0 + out_full // 2.0)
                    cimg2 = sitk.Compose(simg3, simg4, simg3 // 2.0 + simg4 // 2.0)

                    aligned_fakeB_array = sitk.GetArrayFromImage(
                        out
                    )  # registered_fakeBtransformed_segmentation
                    aligned_realB_array_full = sitk.GetArrayFromImage(fixed_full)
                    
                    composition = sitk.GetArrayFromImage(cimg)
                    composition2 = sitk.GetArrayFromImage(cimg2)"""
                    aligned_fakeB_array_full = sitk.GetArrayFromImage(out_full)
                    """print(
                        "aligned fakeB",
                        aligned_fakeB_array_full.dtype,
                        aligned_fakeB_array_full.min(),
                        aligned_fakeB_array_full.max(),
                        aligned_fakeB_array_full.shape,
                    )"""
                    aligned_fakeB_array_full_t = fakeB.new_tensor(
                        [[np.array(aligned_fakeB_array_full)]]
                    )  # ,dtype= float,device='cuda:0')
                    # aligned_fakeB_array_full_t = aligned_fakeB_array_full_t.unsqueeze(0).unsqueeze(0)
                    """aligned_realB_array_full_t = realB.new_tensor(
                        [[np.array(aligned_realB_array_full)]]
                    ) """  # ,dtype= float,device='cuda:0')
                    # aligned_realB_array_full_t = aligned_realB_array_full_t.unsqueeze(0).unsqueeze(0)
                    """print(
                        "aligned torch",
                        aligned_fakeB_array_full_t.dtype,
                        aligned_fakeB_array_full_t.shape,
                        "real B no process",
                        sliding_stack_realB_t.dtype,
                        sliding_stack_realB_t.shape,  # aligned_realB_array_full_t.shape,
                    )"""
                    for slice in range(0, aligned_fakeB_array_full.shape[0]):
                        zero_fraction = np.mean(
                            aligned_fakeB_array_full[slice, :, :] == -601
                        )
                        if zero_fraction <= 0.05:
                            metrics_reg = calculate_metrics(
                                aligned_fakeB_array_full_t[:, :, slice, :, :],  #
                                sliding_stack_realB_t[
                                    :, :, slice, :, :
                                ],  # aligned_realB_array_full_t[:, :, slice, :, :],  #
                            )
                            if result_reg:
                                """print(result_fold_reg.keys())
                                if result_fold_reg[stacked_name_file[slice]]:#stacked_name_file[slice] in result_fold_reg and not
                                    print('already included', result_fold_reg[stacked_name_file[slice]])
                                    continue
                                else:"""
                                result_fold_reg[stacked_name_file[slice]] = metrics_reg
                            """print(
                                stacked_name_file[slice],
                                result_fold_reg[stacked_name_file[slice]],
                            )"""
                        else:
                            print(
                                "Error fraction:",
                                stacked_name_file[slice],
                                zero_fraction,
                            )
                        """plt.figure(figsize=(15, 10))

                        plt.subplot(2, 3, 1)
                        plt.imshow(sliding_stack_realB[0, 0, slice, :, :], cmap="gray")
                        plt.title("Original RealB")
                        plt.axis("off")

                        plt.subplot(2, 3, 2)
                        plt.imshow(
                            composition[slice, :, :], cmap="gray"
                        )  # sliding_stack_fakeB[0,0,slice,:, :]
                        plt.title("Difference")
                        plt.axis("off")

                        plt.subplot(2, 3, 3)
                        plt.imshow(aligned_fakeB_array[slice, :, :], cmap="gray")
                        plt.title("Aligned FakeB")
                        plt.axis("off")

                        plt.subplot(2, 3, 4)
                        plt.imshow(sliding_stack_realB[0, 0, slice, :, :], cmap="gray")
                        plt.title("Original RealB")
                        plt.axis("off")

                        plt.subplot(2, 3, 5)
                        plt.imshow(
                            composition2[slice, :, :], cmap="gray"
                        )  # sliding_stack_fakeB[0,0,slice,:, :]
                        plt.title("Difference")
                        plt.axis("off")

                        plt.subplot(2, 3, 6)
                        plt.imshow(aligned_fakeB_array_full[slice, :, :], cmap="gray")
                        plt.title("Aligned FakeB")
                        plt.axis("off")

                        plt.tight_layout()
                        plt.show()"""
                    stacked_fakeBs = stacked_fakeBs[-2:]
                    stacked_realBs = stacked_realBs[-2:]
                    stacked_fakeBs_bone = stacked_fakeBs_bone[-2:]
                    stacked_realBs_bone = stacked_realBs_bone[-2:]
                    stacked_name_file = stacked_name_file[-2:]

                    """import plotly.graph_objs as go
                    from plotly.subplots import make_subplots

                    # Assuming aligned_fakeB_array is the NumPy array of the aligned fakeB image
                    # Assuming sliding_stack_realB is the original realB image

                    # Create Plotly figure with subplots
                    fig = make_subplots(rows=1, cols=2, subplot_titles=('Original RealB', 'Aligned FakeB'))

                    # Create traces for the original realB and aligned fakeB
                    trace_realB = go.Image(sliding_stack_realB[0,0,0,:,:])  # Assuming it's a 3D image stack, selecting the first slice
                    trace_aligned_fakeB = go.Image(aligned_fakeB_array[0,:, :])  # Assuming it's a 3D image stack, selecting the first slice

                    # Add traces to subplots
                    fig.add_trace(trace_realB, row=1, col=1)
                    fig.add_trace(trace_aligned_fakeB, row=1, col=2)

                    # Update layout
                    fig.update_layout(title='Registration Result: RealB and Aligned FakeB', height=600, width=1000)

                    # Show plot
                    fig.show()

                    test= calculate_metrics(aligned_fakeB_array,sliding_stack_realB)#[0,:,:,:,:][0,:,:,:,:]
                    print('Metrics sliding', test )"""

                metrics = calculate_metrics(fakeB, realB)
                # print("Sans recalage", data_name, metrics)
                if metrics:
                    result_fold[data_name] = metrics
                    # print("before mv process", result_fold[data_name])
                    result_fold_mv[data_name] = calculate_metrics(realA, realB)
                    # print("MV metrics", result_fold_mv[data_name])
                    result_fold_ana[data_name] = calculate_metrics(recB, realB)
                    # print("Ana kv", result_fold_ana[data_name])
                    result_fold_ana_mv[data_name] = calculate_metrics(fakeA, realB)
                    # print("Ana MV", result_fold_ana_mv[data_name])
                else:
                    print("metrics empty", data_name)
                # log_file.write(", %s" % state.metrics)  # save the metrics values
                img_path = model.get_image_paths()  # get image paths
                if i % 5 == 0:  # save images to an HTML file
                    print("processing (%04d)-th image... %s" % (i, img_path))
                if 1000 < i < 1200 or 5000 < i < 5200 or 10000 < i < 10200:
                    save_images(
                        webpage,
                        visuals,
                        img_path,
                        aspect_ratio=opt.aspect_ratio,
                        width=opt.display_winsize,
                        use_wandb=opt.use_wandb,
                    )

            result[k].append(result_fold)
            result_mv[k].append(result_fold_mv)
            result_ana[k].append(result_fold_ana)
            result_ana_mv[k].append(result_fold_ana_mv)
            result_reg[k].append(result_fold_reg)
            webpage.save()  # save the HTML
        log_name = os.path.join(web_dir, "metric_log_fold.json")
        with open(log_name, "w") as fp:
            json.dump(result, fp, indent=4, sort_keys=True, default=str)
        log_name_mv = os.path.join(web_dir, "metric_mv_log_fold.json")
        with open(log_name_mv, "w") as fp:
            json.dump(result_mv, fp, indent=4, sort_keys=True, default=str)
        log_name_ana = os.path.join(web_dir, "metric_ana_log_fold.json")
        with open(log_name_ana, "w") as fp:
            json.dump(result_ana, fp, indent=4, sort_keys=True, default=str)
        log_name_ana_mv = os.path.join(web_dir, "metric_ana_mv_log_fold.json")
        with open(log_name_ana_mv, "w") as fp:
            json.dump(result_ana_mv, fp, indent=4, sort_keys=True, default=str)
        log_name_reg = os.path.join(web_dir, "metric_reg_log_fold.json")
        with open(log_name_reg, "w") as fp:
            json.dump(result_reg, fp, indent=4, sort_keys=True, default=str)
        # Write the results to a JSON file
        log_name = os.path.join(web_dir, "metric_means_by_k.json")
        log_name_mv = os.path.join(web_dir, "metric_mv_means_by_k.json")

        with open(log_name, "w") as file:
            json.dump(
                rank_data_by_metrics(calculate_mean_metrics(result)), file, indent=4
            )
        with open(log_name_mv, "w") as file:
            json.dump(calculate_mean_metrics(result_mv), file, indent=4)
        log_name_ana = os.path.join(web_dir, "metric_ana_means_by_k.json")
        with open(log_name_ana, "w") as file:
            json.dump(
                rank_data_by_metrics(calculate_mean_metrics(result_ana)), file, indent=4
            )
        log_name_ana_mv = os.path.join(web_dir, "metric_ana_mv_means_by_k.json")
        with open(log_name_ana_mv, "w") as file:
            json.dump(calculate_mean_metrics(result_ana_mv), file, indent=4)
        log_name_reg = os.path.join(web_dir, "metric_reg_means_by_k.json")
        with open(log_name_reg, "w") as file:
            json.dump(calculate_mean_metrics(result_reg), file, indent=4)
        print("Results have been written to 'metric_log_fold.json'.")
