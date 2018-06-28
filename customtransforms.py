from __future__ import division

import random
from PIL import Image
import math
import os


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotation(object):
    """Randomly rotate the given PIL.Image. Probability of 0.25 for each rotation.
    """

    def __call__(self, img):
        rand = random.random()
        if rand < 0.25:
            return img.transpose(Image.ROTATE_90)
        elif rand < 0.5:
            return img.transpose(Image.ROTATE_180)
        elif rand < 0.75:
            return img.transpose(Image.ROTATE_270)
        return img
