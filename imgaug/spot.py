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
from skimage import draw
from skimage import data
from skimage import io
from skimage import util
import six.moves as sm
import random

from .meta import Augmenter

class Spot(Augmenter): # pylint: disable=locally-disabled, unused-variable, line-too-long

    def __init__(self, sigma=0, name=None, deterministic=False, random_state=None):
        super(Spot, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

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
        def spots(img,x,y,r):
            opacity = 0.3
            while r>0:
                rr, cc = draw.circle(x,y,r)
                draw.set_color(img,[rr,cc],[255,255,255],alpha=opacity)
                r -= 1
                opacity += 0.2
                if opacity>1:
                    opacity = 1
        for i in sm.xrange(nb_images):
            sig = samples[i]
            if sig > 0 + self.eps:
                for num in range(0,10):
                    r = sig*random.random()
                    x = random.randint(10,images[i].shape[0]-11)
                    y = random.randint(10,images[i].shape[1]-11)
                    spots(images[i],x,y,r)
                    r2 = r-1
                    x2 = random.randint(int(x-sig/2),int(x+sig/2))
                    y2 = random.randint(int(y-sig/2),int(y+sig/2))
                    spots(images[i],x2,y2,r2)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.sigma]