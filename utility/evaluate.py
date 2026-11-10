import math
import numpy as np
import PIL.Image as Image
import os
import cv2
import skimage

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from ntpath import basename  # 用于获取文件名
from os.path import join  # 用于拼接路径
from utility.uiqm_utils import getUIQM
from glob import glob
from scipy.ndimage import gaussian_filter


def getSSIM(X, Y):
    """
       Computes the mean structural similarity between two images.
    """
    assert (X.shape == Y.shape), "Image-patche provided have different dimensions"
    nch = 1 if X.ndim == 2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[..., ch].astype(np.float64), Y[..., ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)


def compute_ssim(X, Y):
    """
       Compute the structural similarity per single channel (given two images)
    """
    # variables are initialized as suggested in the paper
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5

    # means
    ux = gaussian_filter(X, sigma)
    uy = gaussian_filter(Y, sigma)

    # variances and covariances
    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    # normalize by unbiased estimate of std dev
    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)  # eq. 4 of the paper
    vx = (uxx - ux * ux) * unbiased_norm
    vy = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # compute SSIM (eq. 13 of the paper)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim / D
    mssim = SSIM.mean()

    return mssim


def getPSNR(X, Y):
    # assume RGB image
    target_data = np.array(X, dtype=np.float64)
    ref_data = np.array(Y, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    mse = np.mean(diff ** 2.)
    if mse == 0:
        return 100
    else:
        return 10 * math.log10(255.0 * 255.0 / mse), mse


def batch_PSNR(img, imclean, data_range):
    """计算批次图像的PSNR值"""
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def getUCIQE(rgb_in):
    # calculate Chroma
    rgb_in = cv2.normalize(rgb_in, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    (l, a, b) = cv2.split(rgb_in)
    Chroma = np.sqrt(a * a + b * b)
    StdVarianceChroma = np.std(np.reshape(Chroma[:, :], (-1, 1)))

    hsv = skimage.color.rgb2hsv(rgb_in)
    Saturation = hsv[:, :, 2]
    MeanSaturation = np.mean(np.reshape(Saturation[:, :], (-1, 1)))

    ContrastLuminance = max(np.reshape(l[:, :], (-1, 1))) - min(np.reshape(l[:, :], (-1, 1)))
    UCIQE = 0.4680 * StdVarianceChroma + 0.2745 * ContrastLuminance + 0.2576 * MeanSaturation
    return float(UCIQE)


def no_reference_evaluation(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uiqms, uciqes = [], []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uiqms.append(uiqm)
        uciqe = getUCIQE(np.array(im))
        uciqes.append(uciqe)
    return np.array(uiqms), np.array(uciqes)


def evaluate_images(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images
    """

    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims, psnrs, mses = [], [], []  # 添加mses列表
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        # assumes same filenames
        r_im = Image.open(gtr_path).resize(im_res)
        g_im = Image.open(gen_path).resize(im_res)

        if (gtr_f == gen_f):
            # 计算RGB通道的MSE
            # get ssim on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)

            psnr, mse = getPSNR(np.array(r_im), np.array(g_im))
            mses.append(mse)

            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs), np.array(mses)

