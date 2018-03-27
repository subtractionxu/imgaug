#usr/bin/python
'''
Xu, Jian 2018.3.27
'''
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, DiscreteUniform, Uniform
import numpy as np
from scipy import ndimage
import six.moves as sm
import random
import cv2

from .meta import Augmenter

class Flare(Augmenter): # pylint: disable=locally-disabled, unused-variable, line-too-long

    def __init__(self, sigma=0, name=None, deterministic=False, random_state=None):
        super(Flare, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(sigma):
            self.sigma = Deterministic(sigma)
        elif ia.is_iterable(sigma):
            ia.do_assert(len(sigma) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(sigma),))
            self.sigma = Uniform(sigma[0], sigma[1])
        elif isinstance(sigma, StochasticParameter):
            self.sigma = sigma
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(sigma),))

        self.eps = 0.001 # epsilon value to estimate whether sigma is above 0

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.sigma.draw_samples((nb_images,), random_state=random_state)
        for n in sm.xrange(nb_images):
            sig = int(samples[n])
            if sig > 1 + self.eps:
                for i in range(1,sig):
                    x0=random.randint(0,images[n].shape[0]-2)
                    x1=random.randint(x0+2,images[n].shape[0])
                    y0=random.randint(0,images[n].shape[1]-2)
                    y1=random.randint(y0+2,images[n].shape[1])
                    tmpimg=images[n][x0:x1, y0:y1]
                    rows, cols, channels = tmpimg.shape
                    dst = tmpimg.copy()
                    a = random.uniform(0.5, 1.5)
                    b = random.uniform(-50, 50)
                    for i in range(rows):
                        for j in range(cols):
                            for c in range(3):
                                if tmpimg[i, j][c] * a + b > 255:
                                    dst[i, j][c] = tmpimg[i, j][c]
                                elif tmpimg[i, j][c] * a + b < 0:
                                    dst[i, j][c] = tmpimg[i, j][c]
                                else :
                                    dst[i, j][c] = tmpimg[i, j][c] * a + b
                    images[n][x0:x1, y0:y1]=dst
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.sigma]