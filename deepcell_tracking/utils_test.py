# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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

import errno
import os
import shutil
import tempfile

import numpy as np
import skimage as sk

import pytest

from deepcell_tracking import utils


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img

def _get_annotated_image(img_size=256, num_labels=3, sequential=True, seed=1):
    np.random.seed(seed)
    im = np.zeros((img_size, img_size))
    points = img_size * np.random.random((2, num_labels))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    im = sk.filters.gaussian(im, sigma=5)
    blobs = im > 0.7 * im.mean()
    all_labels, num_labels_act = sk.measure.label(blobs, return_num=True)
    assert num_labels == num_labels_act, 'Labels have merged. Increase image ' \
                                         'size or reduce the number of labels'

    if not sequential:
        labels_in_frame = np.unique(all_labels)
        for label in range(num_labels):
            curr_label = label + 1
            new_label = np.random.randint(1, num_labels * 100)
            while new_label in labels_in_frame:
                new_label = np.random.randint(1, num_labels * 100)
            labels_in_frame = np.append(labels_in_frame, new_label)
            label_loc = np.where(all_labels == curr_label)
            all_labels[:, :][label_loc] = new_label

    return all_labels.astype('int32')

def _get_annotated_movie(img_size=256, labels_per_frame=3, frames=3,
                         mov_type='sequential', seed=1,
                         data_format='channels_last'):
    if mov_type in ('sequential', 'repeated'):
        sequential = True
    elif mov_type == 'random':
        sequential = False
    else:
        raise ValueError('mov_type must be one of "sequential", '
                         '"repeated" or "random"')

    if data_format == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 0

    y = []
    while len(y) < frames:
        _y = _get_annotated_image(img_size=img_size, num_labels=labels_per_frame,
                                  sequential=sequential, seed=seed)
        y.append(_y)
        seed += 1

    y = np.stack(y, axis=0)  # expand to 3D

    if mov_type == 'sequential':
        for frame in range(frames):
            if frame == 0:
                new_label = labels_per_frame
                continue
            for label in range(labels_per_frame):
                curr_label = label + 1
                new_label += 1
                label_loc = np.where(y[frame, :, :] == curr_label)
                y[frame, :, :][label_loc] = new_label

    y = np.expand_dims(y, axis=channel_axis)

    return y.astype('int32')


class TestTrackingUtils(object):

    def test_clean_up_annotations(self):
        img = sk.measure.label(sk.data.binary_blobs(length=256, n_dim=2)) * 3
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)  # time axis
        uid = 1

        cleaned = utils.clean_up_annotations(
            img, uid=uid, data_format='channels_last')
        unique = np.unique(cleaned)
        assert len(np.unique(img)) == len(unique)
        expected = np.arange(len(unique)) + uid - 1
        expected[0] = 0  # background shouldn't get added
        np.testing.assert_equal(expected, unique)

        img = sk.measure.label(sk.data.binary_blobs(length=256, n_dim=2)) * 3
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=1)  # time axis

        cleaned = utils.clean_up_annotations(
            img, uid=uid, data_format='channels_first')
        unique = np.unique(cleaned)
        assert len(np.unique(img)) == len(unique)
        expected = np.arange(len(unique)) + uid - 1
        expected[0] = 0  # background shouldn't get added
        np.testing.assert_equal(expected, unique)

        # Correctness check
        for mov_type in ('random', 'repeated'):
            labels_per_frame = 3
            frames = 3
            movie = _get_annotated_movie(img_size=256,
                                         labels_per_frame=labels_per_frame,
                                         frames=frames,
                                         mov_type=mov_type, seed=1,
                                         data_format='channels_last')
            cleaned = utils.clean_up_annotations(movie, uid=uid)
            for frame in range(frames):
                unique = np.unique(cleaned[frame, :, :, 0])
                start = (frame * labels_per_frame) + 1
                end = labels_per_frame * (frame + 1)
                expected = np.arange(start, end + 1, 1, dtype='int32')
                if start != 0:
                    expected = np.append(0, expected)
                np.testing.assert_array_equal(unique, expected)

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

    def test_save_trks(self):
        X = _get_image(30, 30)
        y = np.random.randint(low=0, high=10, size=X.shape)
        lineage = [dict()]

        try:
            tempdir = tempfile.mkdtemp()  # create dir
            with pytest.raises(ValueError):
                badfilename = os.path.join(tempdir, 'x.trk')
                utils.save_trks(badfilename, lineage, X, y)

            filename = os.path.join(tempdir, 'x.trks')
            utils.save_trks(filename, lineage, X, y)
            assert os.path.isfile(filename)

            # test saved tracks can be loaded
            loaded = utils.load_trks(filename)
            assert loaded['lineages'] == lineage
            np.testing.assert_array_equal(X, loaded['X'])
            np.testing.assert_array_equal(y, loaded['y'])

        finally:
            try:
                shutil.rmtree(tempdir)  # delete directory
            except OSError as exc:
                if exc.errno != errno.ENOENT:  # no such file or directory
                    raise  # re-raise exception

    def test_get_max_cells(self):
        labels_per_frame = 5
        frames = 5
        expected_max = labels_per_frame * frames
        y = _get_annotated_movie(img_size=256,
                                 labels_per_frame=labels_per_frame,
                                 frames=frames,
                                 mov_type='sequential', seed=1,
                                 data_format='channels_last')
        calculated_max = utils.get_max_cells(y)
        assert expected_max == calculated_max
