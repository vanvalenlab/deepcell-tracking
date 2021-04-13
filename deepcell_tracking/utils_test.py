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

import copy
import errno
import os
import shutil
import tempfile

import numpy as np
import skimage as sk

import pytest

from deepcell_tracking import utils
from deepcell_tracking.test_utils import _get_image
from deepcell_tracking.test_utils import _get_annotated_image
from deepcell_tracking.test_utils import _get_annotated_movie


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
        frames = 2
        expected_max = labels_per_frame * 2
        y1 = _get_annotated_movie(img_size=256,
                                  labels_per_frame=labels_per_frame,
                                  frames=frames,
                                  mov_type='sequential', seed=1,
                                  data_format='channels_last')
        y2 = _get_annotated_movie(img_size=256,
                                  labels_per_frame=labels_per_frame * 2,
                                  frames=frames,
                                  mov_type='sequential', seed=2,
                                  data_format='channels_last')
        y3 = _get_annotated_movie(img_size=256,
                                  labels_per_frame=labels_per_frame,
                                  frames=frames,
                                  mov_type='sequential', seed=3,
                                  data_format='channels_last')
        y = np.concatenate((y1, y2, y3))
        calculated_max = utils.get_max_cells(y)
        assert expected_max == calculated_max

    def test_relabel_sequential_lineage(self):
        # create dummy movie
        image1 = _get_annotated_image(num_labels=1, sequential=False)
        image2 = _get_annotated_image(num_labels=2, sequential=False)
        movie = np.stack([image1, image2], axis=0)

        # create dummy lineage
        lineage = {}
        for frame in range(movie.shape[0]):
            unique_labels = np.unique(movie[frame])
            unique_labels = unique_labels[unique_labels != 0]
            for unique_label in unique_labels:
                lineage[unique_label] = {
                    'frames': [frame],
                    'parent': None,
                    'daughters': [],
                    'label': unique_label,
                }
        # assign parents and daughters
        parent_label = np.unique(movie[0])
        parent_label = parent_label[parent_label != 0][0]

        daughter_labels = np.unique(movie[1])
        daughter_labels = [d for d in daughter_labels if d]

        lineage[parent_label]['daughters'] = daughter_labels
        for d in daughter_labels:
            lineage[d]['parent'] = parent_label

        # relabel the movie and lineage
        new_movie, new_lineage = utils.relabel_sequential_lineage(movie, lineage)
        new_parent_label = int(np.unique(new_movie[np.where(movie == parent_label)]))

        # test parent is relabeled
        assert new_parent_label == 1  # sequential should start at 1
        assert new_lineage[new_parent_label]['frames'] == lineage[parent_label]['frames']
        assert new_lineage[new_parent_label]['parent'] is None
        assert new_lineage[new_parent_label]['label'] == new_parent_label

        # test daughters are relabeled
        new_daughter_labels = new_lineage[new_parent_label]['daughters']
        assert len(new_daughter_labels) == 2

        for d in new_daughter_labels:
            old_label = int(np.unique(movie[np.where(new_movie == d)]))
            assert new_lineage[d]['frames'] == lineage[old_label]['frames']
            assert new_lineage[d]['parent'] == new_parent_label
            assert new_lineage[d]['label'] == d
            assert not new_lineage[d]['daughters']

    def test_is_valid_lineage(self):
        lineage = {
            0: {'frames': [0],
                'daughters': [1, 2],
                'capped': True,
                'frame_div': 1,
                'parent': None},
            1: {'frames': [1],
                'daughters': [],
                'capped': False,
                'frame_div': 1,
                'parent': 0},
            2: {'frames': [1],
                'daughters': [],
                'capped': False,
                'frame_div': 1,
                'parent': 0,
            }
        }
        assert utils.is_valid_lineage(lineage)

        # change cell 2's daughter frame to 2, should fail
        bad_lineage = copy.copy(lineage)
        bad_lineage[2]['frames'] = [2]
        assert not utils.is_valid_lineage(bad_lineage)
