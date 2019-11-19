# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tracking/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tracking_utils"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import skimage as sk

from deepcell_tracking import utils


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


class TestTrackingUtils(object):

    def test_clean_up_annotations(self):
        img = sk.measure.label(sk.data.binary_blobs(length=256, n_dim=2)) * 3
        img = np.expand_dims(img, axis=-1)
        uid = 100

        cleaned = utils.clean_up_annotations(
            img, uid=uid, data_format='channels_last')
        unique = np.unique(cleaned)
        assert len(np.unique(img)) == len(unique)
        expected = np.arange(len(unique)) + uid - 1
        expected[0] = 0  # background shouldn't get added
        np.testing.assert_equal(expected, unique)

        img = sk.measure.label(sk.data.binary_blobs(length=256, n_dim=2)) * 3
        img = np.expand_dims(img, axis=0)

        cleaned = utils.clean_up_annotations(
            img, uid=uid, data_format='channels_first')
        unique = np.unique(cleaned)
        assert len(np.unique(img)) == len(unique)
        expected = np.arange(len(unique)) + uid - 1
        expected[0] = 0  # background shouldn't get added
        np.testing.assert_equal(expected, unique)

    def test_resize(self):
        channel_sizes = (3, 1)  # skimage used for multi-channel, cv2 otherwise
        for c in channel_sizes:
            for data_format in ('channels_last', 'channels_first'):
                channel_axis = 2 if data_format == 'channels_last' else 0
                img = np.stack([_get_image()] * c, axis=channel_axis)

                resize_shape = (28, 28)
                resized_img = utils.resize(img, resize_shape,
                                           data_format=data_format)

                if data_format == 'channels_first':
                    assert resized_img.shape[1:] == resize_shape
                else:
                    assert resized_img.shape[:-1] == resize_shape

    def test_count_pairs(self):
        batches = 1
        frames = 2
        classes = 4
        prob = 0.5
        expected = batches * frames * classes * (classes + 1) / prob

        # channels_last
        y = np.random.randint(low=0, high=classes + 1,
                              size=(batches, frames, 30, 30, 1))
        pairs = utils.count_pairs(y, same_probability=prob)
        assert pairs == expected

        # channels_first
        y = np.random.randint(low=0, high=classes + 1,
                              size=(batches, 1, frames, 30, 30))
        pairs = utils.count_pairs(
            y, same_probability=prob, data_format='channels_first')
        assert pairs == expected
