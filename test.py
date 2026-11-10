import torch
import os
import warnings
import numpy as np

warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)

from model import AAAF
from utility.evaluate import *
from utility.image_process import process_image
from utility.init import init_test
from config import cfg


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_default_tensor_type(torch.FloatTensor)


def main_all(config, model):

    generator, model_files = init_test(config, model)

    records = []

    print(f"\n{'=' * 85}")
    print(f"{'Epoch':<8} | {'PSNR':<10} | {'SSIM':<10} | {'MSE(x10³)':<10} | {'UIQM':<10} | {'UCIQE':<10} | {'FPS':<8}")
    print(f"{'-' * 85}")

    for model_file in model_files:
        epoch = int(model_file.split("_")[1].split(".")[0])

        generator.load_state_dict(torch.load(os.path.join(config.saved_models_path, model_file)))
        generator.half()
        generator.eval()

        test_files = sorted(os.listdir(config.test_raw_path), key=lambda x: int(x.split(".")[0]))
        times = []
        t = config.test_raw_path
        for filename in test_files:
            input_path = os.path.join(t, filename)
            output_path = os.path.join(config.output_path, filename)
            times.append(process_image(input_path, output_path, generator))

        re =  config.output_path
        # ssim, psnr, mse = evaluate_images(config.test_gt_path,re)

        # uiqm ,uciqe = no_reference_evaluation(re)
        # uiqm_mean = np.mean(uiqm)
        # uciqe_mean = np.mean(uciqe)
        #
        # psnr_mean = np.mean(psnr)
        # ssim_mean = np.mean(ssim)
        # mse_mean = np.mean(mse)
        fps_val = 1.0 / np.mean(times[2:])
        print(fps_val)
    #     records.append({
    #         'epoch': epoch,
    #         'psnr': psnr_mean,
    #         'ssim': ssim_mean,
    #         'mse': mse_mean,
    #         'uiqm': uiqm_mean,
    #         "uciqe": uciqe_mean,
    #         'fps': fps_val
    #     })
    #
    #     print(
    #         f"{epoch:<8} | {psnr_mean:<10.4f} | {ssim_mean:<10.4f} | {mse_mean / 1000:<10.4f} | {uiqm_mean :<10.4f} | {uciqe_mean :<10.4f} | {fps_val:<8.2f} |")
    #
    #
    # if records:
    #     print(f"{'=' * 85}")
    #     print(f"\n>>> 最佳指标汇总:")
    #     print(f"{'指标':<10} | {'最佳数值':<12} | {'对应轮数 (Epoch)':<15}")
    #     print("-" * 45)
    #
    #     # PSNR
    #     best_psnr = max(records, key=lambda x: x['psnr'])
    #     print(f"{'PSNR':<10} | {best_psnr['psnr']:<12.4f} | {best_psnr['epoch']:<15}")
    #
    #     # SSIM
    #     best_ssim = max(records, key=lambda x: x['ssim'])
    #     print(f"{'SSIM':<10} | {best_ssim['ssim']:<12.4f} | {best_ssim['epoch']:<15}")
    #
    #     # MSE
    #     best_mse = min(records, key=lambda x: x['mse'])
    #     print(f"{'MSE':<10} | {best_mse['mse'] / 1000:<12.4f} | {best_mse['epoch']:<15}")
    #
    #     # UIQM
    #     best_uiqm = max(records, key=lambda x: x['uiqm'])
    #     print(f"{'UIQM':<10} | {best_uiqm['uiqm']:<12.4f} | {best_uiqm['epoch']:<15}")
    #
    #     # UCIQE
    #     best_uciqe = max(records, key=lambda x: x['uciqe'])
    #     print(f"{'UCIQE':<10} | {best_uciqe['uciqe']:<12.4f} | {best_uciqe['epoch']:<15}")

        # FPS
        best_fps = max(records, key=lambda x: x['fps'])
        print(f"{'FPS':<10} | {best_fps['fps']:<12.2f} | {best_fps['epoch']:<15}")

        print("=" * 45)


def testing(dataset):
    config = cfg(dataset)
    main_all(config, AAAF)


if __name__ == "__main__":
    testing("UIEB")