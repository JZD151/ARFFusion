import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from skimage.metrics import structural_similarity as ssim
from pyiqa import create_metric
import torch
import os
from dataset import fusion_dataset, fusion_dataset_gt
from rgb2ycbcr import RGB2YCrCb, YCrCb2RGB

import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from rgb2ycbcr import RGB2YCrCb, YCrCb2RGB

class Evaluator:
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):
        if imgA is None:
            assert isinstance(imgF, np.ndarray), 'type error'
            assert imgF.ndim == 2, 'dimension error'
        else:
            assert isinstance(imgF, np.ndarray) and isinstance(imgA, np.ndarray) and isinstance(imgB, np.ndarray), 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert imgF.ndim == 2, 'dimension error'

    @classmethod
    def EN(cls, img):
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a, minlength=256) / a.size
        return -np.sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        mi_FA = skm.mutual_info_score(image_F.flatten(), image_A.flatten())
        mi_FB = skm.mutual_info_score(image_F.flatten(), image_B.flatten())
        return mi_FA + mi_FB

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        def corr(x, y):
            xm, ym = x - x.mean(), y - y.mean()
            return np.sum(xm * ym) / np.sqrt(np.sum(xm**2) * np.sum(ym**2) + 1e-10)
        return 0.5 * (corr(image_A, image_F) + corr(image_B, image_F))

    @classmethod
    def SCD(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        diff_A = image_F - image_A
        diff_B = image_F - image_B
        def corr(x, y):
            xm, ym = x - x.mean(), y - y.mean()
            return np.sum(xm * ym) / np.sqrt(np.sum(xm**2) * np.sum(ym**2) + 1e-10)
        return corr(image_A, diff_B) + corr(image_B, diff_A)

    @classmethod
    def VIF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F) + cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls, ref, dist):
        sigma_nsq = 2
        eps = 1e-10
        num = den = 0.0
        for scale in range(1, 5):
            N = 2**(5 - scale) + 1
            sd = N / 5.0
            m = (N - 1) / 2.0
            y, x = np.ogrid[-m:m+1, -m:m+1]
            kernel = np.exp(-(x*x + y*y) / (2 * sd * sd))
            kernel[kernel < eps * kernel.max()] = 0
            win = kernel / kernel.sum() if kernel.sum() != 0 else kernel

            if scale > 1:
                ref = convolve2d(ref, win, mode='valid')[::2, ::2]
                dist = convolve2d(dist, win, mode='valid')[::2, ::2]

            mu_ref = convolve2d(ref, win, mode='valid')
            mu_dist = convolve2d(dist, win, mode='valid')
            sigma_ref_sq = np.maximum(convolve2d(ref*ref, win, mode='valid') - mu_ref**2, 0)
            sigma_dist_sq = np.maximum(convolve2d(dist*dist, win, mode='valid') - mu_dist**2, 0)
            sigma_ref_dist = convolve2d(ref*dist, win, mode='valid') - mu_ref * mu_dist

            g = sigma_ref_dist / (sigma_ref_sq + eps)
            sv_sq = sigma_dist_sq - g * sigma_ref_dist
            g = np.where(sigma_ref_sq < eps, 0, g)
            sv_sq = np.where(sigma_ref_sq < eps, sigma_dist_sq, sv_sq)
            g = np.clip(g, 0, None)
            sv_sq = np.maximum(sv_sq, eps)

            num += np.sum(np.log10(1 + (g**2) * sigma_ref_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma_ref_sq / sigma_nsq))

        vif = num / den if den != 0 else 1.0
        return vif if not np.isnan(vif) else 1.0


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode in ('RGB', 'GRAY', 'YCrCb'), 'mode error'
    if mode == 'RGB':
        return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    if mode == 'GRAY':
        return np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)


def evaluate(vis_path, ir_path, fus_path, logger=None, description="none"):
    test_dataset = fusion_dataset_gt(vis_path=vis_path, ir_path=ir_path, gt_path=fus_path, model='test')
    metric_result = np.zeros(6)

    for vi, ir, fi, name in test_dataset:
        # convert to Y channel if RGB
        if vi.shape[-3] == 3:
            vi = RGB2YCrCb(vi.unsqueeze(0))[0, :1, :, :]
        if fi.shape[-3] == 3:
            fi = RGB2YCrCb(fi.unsqueeze(0))[0, :1, :, :]

        vi = (vi[0].cpu().numpy() * 255).round()
        ir = (ir[0].cpu().numpy() * 255).round()
        fi = (fi[0].cpu().numpy() * 255).round()

        metric_result[0] += Evaluator.EN(fi)
        metric_result[1] += Evaluator.SD(fi)
        metric_result[2] += Evaluator.MI(fi, ir, vi)
        metric_result[3] += Evaluator.CC(fi, ir, vi)
        metric_result[4] += Evaluator.SCD(fi, ir, vi)
        metric_result[5] += Evaluator.VIF(fi, ir, vi)

    metric_result /= len(test_dataset)

    labels = ['EN', 'SD', 'MI', 'CC', 'SCD', 'VIF']
    result_str = description + "\t" + "\t".join(f"{np.round(v,3)}" for v in metric_result)

    if logger:
        logger.info("="*60)
        logger.info("\t" + "\t".join(labels))
        logger.info(result_str)
        logger.info("="*60)
    else:
        print("="*60)
        print("\t" + "\t".join(labels))
        print(result_str)
        print("="*60)
