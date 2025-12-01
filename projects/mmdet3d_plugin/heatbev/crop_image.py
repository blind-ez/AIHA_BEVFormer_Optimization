import random

import numpy as np
from mmdet.datasets.builder import PIPELINES
from PIL import Image

from projects.mmdet3d_plugin.datasets.pipelines.augmentation import CropResizeFlipImage

@PIPELINES.register_module()
class CropImage(CropResizeFlipImage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, results):
        crop, resize, resize_dims = self.sample_augmentation()

        new_imgs = list()
        for cam, img in enumerate(results["img"]):
            new_img = Image.fromarray(np.uint8(img))
            new_img, ida_mat = self.img_transform(new_img, crop, resize, resize_dims)
            results["img"][cam] = np.array(new_img).astype(np.float32)
            results['lidar2img'][cam][:3, :] = np.matmul(ida_mat, results['lidar2img'][cam][:3, :])

        return results

    def img_transform(self, img, crop, resize, resize_dims):
        img = img.crop(crop)
        img = img.resize(resize_dims)

        ida_rot = np.eye(2)
        ida_rot *= resize

        ida_tran = np.zeros(2)
        ida_tran -= np.array(crop[:2]) * resize

        ida_mat = np.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        return img, ida_mat

    def sample_augmentation(self):
        crop = self.data_aug_conf["crop"]

        resized_h = self.data_aug_conf["resize"]
        resized_w = resized_h / (crop[3] - crop[1]) * (crop[2] - crop[0])
        resize = resized_h / (crop[3] - crop[1])
        resize_dims = (int(resized_w), int(resized_h))

        return crop, resize, resize_dims
