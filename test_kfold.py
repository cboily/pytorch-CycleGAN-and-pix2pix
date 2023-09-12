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

# from torchmetrics import MeanAbsoluteError
import torch
from ignite.engine import Engine
from ignite.metrics import PSNR, SSIM, MeanAbsoluteError, RootMeanSquaredError
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


def calculate_mean_metrics(data_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculates the mean of each metric from a list of dictionaries."""

    results = []

    # For each sublist in the data:
    for sublist in data:
        total_mae, total_psnr, total_rmse, total_ssim = 0, 0, 0, 0

        # Sum up the metrics for each file in the sublist in a single pass
        for test_image in sublist:
            for metrics in test_image.values():
                total_mae += metrics["MAE"]
                total_psnr += metrics["PSNR"]
                total_rmse += metrics["RMSE"]
                total_ssim += metrics["SSIM"]

            num_items = len(test_image)
            # Store the mean for each metric in the results list using conventional keys
            sublist_result = {
                "mean_MAE": total_mae / num_items,
                "mean_PSNR": total_psnr / num_items,
                "mean_RMSE": total_rmse / num_items,
                "mean_SSIM": total_ssim / num_items,
            }
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


with torch.no_grad():
    if __name__ == "__main__":
        opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        opt.num_threads = 0  # test code only supports num_threads = 0
        opt.batch_size = 1  # test code only supports batch_size = 1
        # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = (
            True  # no flip; comment this line if results on flipped images are needed.
        )
        opt.display_id = (
            -1
        )  # no visdom display; the test code saves the results to a HTML file.

        opt.num_folds = 5
        opt.seed = 53493403
        opt.isTrain = False
        result = []
        for k in range(0, opt.num_folds):
            result.append([])
            opt.fold = k
            print(f"FOLD {k}")
            print("--------------------------------")
            opt.phase = "train"
            dataset = create_dataset(
                opt
            )  # create a dataset given opt.dataset_mode and other options
            # opt.serial_batches = False
            print("Size Test subset", opt.num_test)
            opt.phase = "test"
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
                "{}_fold_{}_{}".format(opt.phase, opt.fold, opt.epoch),
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
            # log_name = os.path.join(opt.results_dir, opt.name, "metric_log_fold_{}.txt".format(str(k)))
            log_name_mv = os.path.join(
                opt.results_dir, opt.name, "metric_mv_log_fold_{}.txt".format(str(k))
            )
            # open(log_name, "w").close()
            open(log_name_mv, "w").close()
            result_fold = {}
            for i, data in enumerate(dataset):
                result_data = {}
                if i >= opt.num_test:  # only apply our model to opt.num_test images.
                    break
                model.set_input(data)  # unpack data from data loader
                data_name = os.path.split(str(data["A_paths"][0]))[1]
                print(data_name)
                result_data
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results

                out_transforms = Compose(
                    [
                        ScaleIntensityRange(
                            a_min=-1.0,
                            a_max=1.0,
                            b_min=-600.0,
                            b_max=400.0,
                            clip=True,
                        ),
                    ]
                )
                fakeB = out_transforms(visuals["fake_B"])
                realB = out_transforms(visuals["real_B"])
                realA = out_transforms(visuals["real_A"])
                print("Fa", fakeB.min(), fakeB.max())

                result_data["MAE"] = mean_absolute_error(fakeB, realB)
                testssim = StructuralSimilarityIndexMeasure().to(device="cuda:0")
                testpsnr = PeakSignalNoiseRatio().to(device="cuda:0")
                testrmse = MeanSquaredError(squared=False).to(device="cuda:0")
                result_data["RMSE"] = testrmse(fakeB, realB).item()
                result_data["SSIM"] = testssim(fakeB, realB).item()
                result_data["PSNR"] = testpsnr(fakeB, realB).item()

                test_mae_mv = mean_absolute_error(realA, realB)
                test_rmse_mv = testrmse(realA, realB)
                test_psnr_mv = testpsnr(realA, realB)
                test_ssim_mv = testssim(realA, realB)

                """# create default evaluator for doctests
                    def eval_step(_, batch):
                        return batch
                default_evaluator = Engine(eval_step)
                mae = MeanAbsoluteError()
                psnr = PSNR(data_range=1000)
                rmse = RootMeanSquaredError()
                ssim = SSIM(data_range=1000)
                mae.attach(default_evaluator, "mae")
                psnr.attach(default_evaluator, "psnr")
                rmse.attach(default_evaluator, "rmse")
                ssim.attach(default_evaluator, "ssim")
                state = default_evaluator.run(
                    [[fakeB, realB]], epoch_length=1, max_epochs=1
                )
                pprint(state.metrics)"""
                result_fold[data_name] = result_data

                """with open(log_name, "a") as log_file:
                    log_file.write(data_name)
                    log_file.write(
                        ", {'mae_torch': %s, 'psnr_torch': %s, 'rmse_torch': %s, 'ssim_torch': %s}\n"
                        % (
                            test_mae.item(),
                            test_psnr.item(),
                            test_rmse.item(),
                            test_ssim.item(),
                        )
                    )"""
                with open(log_name_mv, "a") as log_file:
                    log_file.write(data_name)
                    log_file.write(
                        ", {'mae_torch': %s, 'psnr_torch': %s, 'rmse_torch': %s, 'ssim_torch': %s}\n"
                        % (
                            test_mae_mv.item(),
                            test_psnr_mv.item(),
                            test_rmse_mv.item(),
                            test_ssim_mv.item(),
                        )
                    )

                    # log_file.write(", %s" % state.metrics)  # save the metrics values
                img_path = model.get_image_paths()  # get image paths
                if i % 5 == 0:  # save images to an HTML file
                    print("processing (%04d)-th image... %s" % (i, img_path))
                save_images(
                    webpage,
                    visuals,
                    img_path,
                    aspect_ratio=opt.aspect_ratio,
                    width=opt.display_winsize,
                    use_wandb=opt.use_wandb,
                )
            result[k].append(result_fold)
            webpage.save()  # save the HTML
        log_name = os.path.join(opt.results_dir, opt.name, "metric_log_fold.json")
        with open(log_name, "w") as fp:
            json.dump(result, fp, indent=4, sort_keys=True, default=str)

        # Write the results to a JSON file
        log_name = os.path.join(opt.results_dir, opt.name, "metrics_means_by_k.json")
        with open(log_name, "w") as file:
            json.dump(
                rank_data_by_metrics(calculate_mean_metrics(data)), file, indent=4
            )

        print("Results have been written to 'results.json'.")
