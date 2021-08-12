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
import os
import io

import numpy as np
import skimage as sk

import pytest

from deepcell_tracking import utils
from deepcell_tracking.test_utils import get_image
from deepcell_tracking.test_utils import get_annotated_image
from deepcell_tracking.test_utils import get_annotated_movie


def get_dummy_data(num_labels=3, batches=2):
    num_labels = 3
    movies = []
    for _ in range(batches):
        movies.append(get_annotated_movie(labels_per_frame=num_labels))

    y = np.stack(movies, axis=0)
    X = np.random.random(y.shape)

    # create dummy lineage
    lineages = {}
    for b in range(X.shape[0]):
        lineages[b] = {}
        for frame in range(X.shape[1]):
            unique_labels = np.unique(y[b, frame])
            unique_labels = unique_labels[unique_labels != 0]
            for unique_label in unique_labels:
                if unique_label not in lineages[b]:
                    lineages[b][unique_label] = {
                        'frames': [frame],
                        'parent': None,
                        'daughters': [],
                        'label': unique_label,
                    }
                else:
                    lineages[b][unique_label]['frames'].append(frame)

    # tracks expect batched data
    data = {'X': X, 'y': y, 'lineages': lineages}
    return data


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
            movie = get_annotated_movie(img_size=256,
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

    def test_save_trks(self, tmpdir):
        X = get_image(30, 30)
        y = np.random.randint(low=0, high=10, size=X.shape)
        lineage = [dict()]

        tempdir = str(tmpdir)
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

        # test save trks to bytes
        b = io.BytesIO()
        utils.save_trks(b, lineage, X, y)

        # load trks from bytes
        b.seek(0)
        loaded = utils.load_trks(b)
        assert loaded['lineages'] == lineage
        np.testing.assert_array_equal(X, loaded['X'])
        np.testing.assert_array_equal(y, loaded['y'])

    def test_save_trk(self, tmpdir):
        X = get_image(30, 30)
        y = np.random.randint(low=0, high=10, size=X.shape)
        lineage = [dict()]

        tempdir = str(tmpdir)
        with pytest.raises(ValueError):
            badfilename = os.path.join(tempdir, 'x.trks')
            utils.save_trk(badfilename, lineage, X, y)

        with pytest.raises(ValueError):
            utils.save_trk('x.trk', [{}, {}], X, y)

        filename = os.path.join(tempdir, 'x.trk')
        utils.save_trk(filename, lineage, X, y)
        assert os.path.isfile(filename)

        # test saved tracks can be loaded
        loaded = utils.load_trks(filename)
        assert loaded['lineages'] == lineage
        np.testing.assert_array_equal(X, loaded['X'])
        np.testing.assert_array_equal(y, loaded['y'])

        # test save trks to bytes
        b = io.BytesIO()
        utils.save_trk(b, lineage, X, y)

        # load trks from bytes
        b.seek(0)
        loaded = utils.load_trks(b)
        assert loaded['lineages'] == lineage
        np.testing.assert_array_equal(X, loaded['X'])
        np.testing.assert_array_equal(y, loaded['y'])

    def test_load_trks(self, tmpdir):
        filename = os.path.join(str(tmpdir), 'bad-lineage.trk')
        X = get_image(30, 30)
        y = np.random.randint(low=0, high=10, size=X.shape)
        lineage = [dict()]

        utils.save_track_data(filename=filename,
                              lineages=lineage,
                              raw=X,
                              tracked=y,
                              lineage_name='bad-lineage.json')

        with pytest.raises(ValueError):
            utils.load_trks(filename)

    def test_normalize_adj_matrix(self):
        frames = 3
        # 2 cells in each frame
        adj = np.zeros((frames, 2, 2))
        for i in range(2):
            adj[:, i, i] = 1

        normalized = utils.normalize_adj_matrix(adj)

        # TODO: check accuracy of normalized tensor.

        # also normalize batches
        batched_adj = np.stack([adj, adj], axis=0)
        batched_normalized = utils.normalize_adj_matrix(batched_adj)

        # each batch is the same, should be normalized consistently
        for b in range(batched_normalized.shape[0]):
            np.testing.assert_array_equal(
                batched_normalized[b],
                normalized)

        # Should fail with too large inputs
        with pytest.raises(ValueError):
            utils.normalize_adj_matrix(np.zeros((32,) * 2))

        # Should fail with too small inputs
        with pytest.raises(ValueError):
            utils.normalize_adj_matrix(np.zeros((32,) * 5))

    def test_get_max_cells(self):
        labels_per_frame = 5
        frames = 2
        expected_max = labels_per_frame * 2
        y1 = get_annotated_movie(img_size=256,
                                 labels_per_frame=labels_per_frame,
                                 frames=frames,
                                 mov_type='sequential', seed=1,
                                 data_format='channels_last')
        y2 = get_annotated_movie(img_size=256,
                                 labels_per_frame=labels_per_frame * 2,
                                 frames=frames,
                                 mov_type='sequential', seed=2,
                                 data_format='channels_last')
        y3 = get_annotated_movie(img_size=256,
                                 labels_per_frame=labels_per_frame,
                                 frames=frames,
                                 mov_type='sequential', seed=3,
                                 data_format='channels_last')
        y = np.concatenate((y1, y2, y3))
        calculated_max = utils.get_max_cells(y)
        assert expected_max == calculated_max

    def test_relabel_sequential_lineage(self):
        # create dummy movie
        image1 = get_annotated_image(num_labels=1, sequential=False)
        image2 = get_annotated_image(num_labels=2, sequential=False)
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
                'parent': 0},
        }
        assert utils.is_valid_lineage(lineage)

        # change cell 2's daughter frame to 2, should fail
        bad_lineage = copy.copy(lineage)
        bad_lineage[2]['frames'] = [2]
        assert not utils.is_valid_lineage(bad_lineage)

    def test_get_image_features(self):
        num_labels = 3
        y = get_annotated_image(num_labels=num_labels, sequential=True)
        y = np.expand_dims(y, axis=-1)
        X = np.random.random(y.shape)

        appearance_dim = 16
        distance_threshold = 64
        features = utils.get_image_features(X, y, appearance_dim)

        # test appearance
        appearances = features['appearances']
        expected_shape = (num_labels, appearance_dim, appearance_dim, X.shape[-1])
        assert appearances.shape == expected_shape

        # test centroids
        centroids = features['centroids']
        expected_shape = (num_labels, 2)
        assert centroids.shape == expected_shape

        # test centroids
        centroids = features['centroids']
        expected_shape = (num_labels, 2)
        assert centroids.shape == expected_shape

        # test morphologies
        morphologies = features['morphologies']
        expected_shape = (num_labels, 3)
        assert morphologies.shape == expected_shape

        # test labels
        labels = features['labels']
        expected_shape = (num_labels,)
        assert labels.shape == expected_shape
        np.testing.assert_array_equal(labels, np.array(list(range(1, num_labels + 1))))

    def test_concat_tracks(self):
        num_labels = 3

        data = get_dummy_data(num_labels)
        track_1 = utils.Track(tracked_data=data)
        track_2 = utils.Track(tracked_data=data)

        data = utils.concat_tracks([track_1, track_2])

        for k, v in data.items():
            starting_batch = 0
            for t in (track_1, track_2):
                assert hasattr(t, k)
                w = getattr(t, k)
                # data is put into top left corner of array
                v_sub = v[
                    starting_batch:starting_batch + w.shape[0],
                    0:w.shape[1],
                    0:w.shape[2],
                    0:w.shape[3]
                ]
                np.testing.assert_array_equal(v_sub, w)

        # test that input must be iterable
        with pytest.raises(TypeError):
            utils.concat_tracks(track_1)

    def test_trks_stats(self):

        # Test bad extension
        with pytest.raises(ValueError):
            utils.trks_stats('bad-extension.npz')


class TestTrack(object):

    def test_init(self, mocker):
        num_labels = 3

        data = get_dummy_data(num_labels)

        # invalidate one lineage
        mocker.patch('deepcell_tracking.utils.load_trks',
                     lambda x: data)

        track1 = utils.Track(tracked_data=data)
        track2 = utils.Track(path='path/to/data')

        np.testing.assert_array_equal(track1.appearances, track2.appearances)
        np.testing.assert_array_equal(
            track1.temporal_adj_matrices,
            track2.temporal_adj_matrices)

        with pytest.raises(ValueError):
            utils.Track()
