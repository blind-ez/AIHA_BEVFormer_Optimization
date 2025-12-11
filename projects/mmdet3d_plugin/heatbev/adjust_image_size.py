import math
import random

import numpy as np
from mmdet.datasets.builder import PIPELINES
from PIL import Image

from projects.mmdet3d_plugin.datasets.pipelines.augmentation import CropResizeFlipImage

@PIPELINES.register_module()
class AdjustImageSize:
    def __init__(self, original_img_size, target_img_size):
        H_ori, W_ori = original_img_size
        H_tgt, W_tgt = target_img_size

        aspect_ori = W_ori / H_ori
        aspect_tgt = W_tgt / H_tgt

        if aspect_ori > aspect_tgt:
            new_w = int(H_ori * aspect_tgt) if float(H_ori * aspect_tgt).is_integer() else math.ceil(H_ori * aspect_tgt)
            new_h = H_ori
            left = (W_ori - new_w) // 2
            top = 0
        else:
            new_w = W_ori
            new_h = int(W_ori / aspect_tgt) if float(W_ori / aspect_tgt).is_integer() else math.ceil(W_ori / aspect_tgt)
            left = 0
            top = (H_ori - new_h) // 2

        self.crop = (left, top, left + new_w, top + new_h)
        self.resize = (W_tgt, H_tgt)

        scale_x = W_tgt / new_w
        scale_y = H_tgt / new_h

        img_transform = np.eye(3)
        img_transform[0, 0] = scale_x
        img_transform[1, 1] = scale_y
        img_transform[0, 2] = -left * scale_x
        img_transform[1, 2] = -top * scale_y

        self.img_transform = img_transform

    def __call__(self, results):
        for cam in range(len(results["img"])):
            new_img = Image.fromarray(np.uint8(results["img"][cam]))
            new_img = new_img.crop(self.crop)
            new_img = new_img.resize(self.resize)
            results["img"][cam] = np.array(new_img).astype(np.float32)
            results['lidar2img'][cam][:3, :] = np.matmul(self.img_transform, results['lidar2img'][cam][:3, :])

        return results
