import torch
import numpy as np
import PIL
import collections
import random

class Scale(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, inputs):
        image, depth, seg = inputs
        oh, ow = self.size
        image = image.resize((ow, oh), PIL.Image.BILINEAR)
        depth = depth.resize((ow, oh), PIL.Image.NEAREST)
        seg = seg.resize((ow, oh), PIL.Image.NEAREST)
        return (image, depth, seg)


class RandomCrop(object):

    def __init__(self, size_image, size_depth):
        self.size_image = size_image
        self.size_depth = size_depth


    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, inputs):
        image, depth, seg = inputs

        i, j, h, w = self.get_params(image, self.size_image)

        image =  image.crop((j, i, j + w, i + h))
        depth = depth.crop((j, i, j + w, i + h))
        seg = seg.crop((j, i, j + w, i + h))
        oh, ow = self.size_depth
        depth = depth.resize((ow, oh), PIL.Image.NEAREST)
        seg = seg.resize((ow, oh), PIL.Image.NEAREST)
        return (image, depth, seg)


