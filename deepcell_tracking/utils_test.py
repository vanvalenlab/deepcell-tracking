# Copyright 2016-2022 The Van Valen Lab at the California Institute of
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

        assert utils.is_valid_lineage(movie, lineage)

        # a cell's frames should match the label array
        bad_lineage = copy.deepcopy(lineage)
        bad_lineage[parent_label]['frames'].append(1)
        assert not utils.is_valid_lineage(movie, bad_lineage)

        # cell in lineage but not in movie is invalid
        max_label = np.max(movie)
        bad_label = np.max(movie) + 1
        bad_lineage = copy.deepcopy(lineage)
        bad_lineage[bad_label] = bad_lineage[max_label]
        assert not utils.is_valid_lineage(movie, bad_lineage)

        # cell in movie but not in lineage is invalid
        bad_movie = copy.deepcopy(movie)
        bad_movie[0, 0, 0] = bad_label
        assert not utils.is_valid_lineage(bad_movie, lineage)

        # a daughter's frames should start immediately
        # after the parent's last frame
        # (not strictly true in data, but required for GNN models)
        bad_frame = 0
        bad_movie = copy.deepcopy(movie)
        bad_movie[bad_frame, 0, 0] = daughter_labels[0]
        bad_lineage = copy.deepcopy(lineage)
        bad_lineage[daughter_labels[0]]['frames'] = [bad_frame]
        assert not utils.is_valid_lineage(bad_movie, bad_lineage)

        # daughter not in lineage is invalid
        bad_lineage = copy.deepcopy(lineage)
        bad_movie = copy.deepcopy(movie)
        daughter_idx = np.where(movie == lineage[parent_label]['daughters'][0])
        bad_movie[daughter_idx] = bad_label
        bad_lineage[parent_label]['daughters'][0] = bad_label
        assert not utils.is_valid_lineage(bad_movie, bad_lineage)

        # daughter not in movie is invalid
        bad_lineage = copy.deepcopy(lineage)
        bad_lineage[parent_label]['daughters'][0] = bad_label
        bad_lineage[bad_label] = bad_lineage[bad_lineage[parent_label]['daughters'][1]]
        assert not utils.is_valid_lineage(movie, bad_lineage)

        # test daughter frames are empty
        bad_lineage = copy.deepcopy(lineage)
        bad_lineage[daughter_labels[0]]['frames'] = []
        assert not utils.is_valid_lineage(movie, bad_lineage)

        # parent ID > daughters ID is OK
        new_parent = max(np.unique(movie)) + 2
        relabeled_movie = np.where(movie == parent_label, new_parent, movie)
        relabeled_lineage = copy.deepcopy(lineage)
        relabeled_lineage[new_parent] = lineage[parent_label]
        del relabeled_lineage[parent_label]
        for daughter in relabeled_lineage[new_parent]['daughters']:
            if relabeled_lineage[daughter]['parent'] == parent_label:
                relabeled_lineage[daughter]['parent'] = new_parent
        assert utils.is_valid_lineage(relabeled_movie, relabeled_lineage)

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

        # test appearance - fixed crop
        features = utils.get_image_features(X, y, appearance_dim,
                                            crop_mode='fixed', norm=True)
        appearances = features['appearances']
        expected_shape = (num_labels, appearance_dim, appearance_dim, X.shape[-1])
        assert appearances.shape == expected_shape

    def test_trks_stats(self):
        # Test bad extension
        with pytest.raises(ValueError):
            utils.trks_stats('bad-extension.npz')

        # No inputs
        with pytest.raises(ValueError):
            utils.trks_stats()

        data = get_dummy_data()
        stats = utils.trks_stats(**data)
        assert isinstance(stats, dict)

    def test_contig_tracks(self):
        # test already contiguous
        frames = 5
        track = {
            1: {
                'label': 1,
                'frames': [0, 1, 2],
                'daughters': [2, 3],
                'parent': None,
            },
            2: {
                'label': 2,
                'frames': [3, 4],
                'daughters': [],
                'parent': 1
            },
            3: {
                'label': 3,
                'frames': [3, 4],
                'daughters': [],
                'parent': 1
            }
        }
        original_track = copy.copy(track)
        original_daughters = original_track[1]['daughters']
        y = np.random.randint(0, 4, size=(frames, 40, 40, 1))
        new_track, _ = utils.contig_tracks(1, track, y)
        assert original_track == new_track

        # test non-contiguous
        track = copy.copy(original_track)
        track[1]['frames'].append(4)
        new_track, _ = utils.contig_tracks(1, track, y)

        assert len(new_track) == len(original_track) + 1
        assert new_track[1]['frames'] == original_track[1]['frames']
        daughters = new_track[max(new_track)]['daughters']
        assert daughters == original_daughters
        for d in daughters:
            assert new_track[d]['parent'] == max(new_track)

    def test_match_nodes(self):
        # creat dummy movie to test against
        labels_per_frame = 5
        frames = 3
        y1 = get_annotated_movie(img_size=256,
                                 labels_per_frame=labels_per_frame,
                                 frames=frames,
                                 mov_type='repeated', seed=1,
                                 data_format='channels_last')
        # test same movie
        gtcells, rescells = utils.match_nodes(y1, y1)
        for gt_cell, res_cell in zip(gtcells, rescells):
            assert gt_cell == res_cell

        # test different movie (with known values)
        y2 = get_annotated_movie(img_size=256,
                                 labels_per_frame=labels_per_frame,
                                 frames=frames,
                                 mov_type='sequential', seed=1,
                                 data_format='channels_last')
        gtcells, rescells = utils.match_nodes(y1, y2)

        assert len(rescells) == len(gtcells)
        for loc, gt_cell in enumerate(np.unique(gtcells)):
            # because movies have the same first frame, every
            # iteration of unique values should match original label
            assert gt_cell == rescells[loc * 3]

    def test_trk_to_graph(self):
        tracks_gt = {1: {'label': 1, 'frames': [1, 2, 3], 'daughters': [],
                         'capped': False, 'frame_div': None, 'parent': 3},
                     2: {'label': 2, 'frames': [1, 2], 'daughters': [],
                         'capped': False, 'frame_div': None, 'parent': 3},
                     3: {'label': 3, 'frames': [0], 'daughters': [1, 2],
                         'capped': False, 'frame_div': 1, 'parent': None},
                     4: {'label': 4, 'frames': [0], 'daughters': [],
                         'capped': False, 'frame_div': None, 'parent': None}
                     }

        G = utils.trk_to_graph(tracks_gt)

        # Calculate possible node ids
        nodes = []
        for id, lin in tracks_gt.items():
            nodes.extend(['{}_{}'.format(id, t) for t in lin['frames']])

        # Check that number of nodes match
        assert len(nodes) == len(G.nodes)

        # Check that all expected nodes are present
        for n in nodes:
            assert n in G

        # Check that division of cell 3 is recorded properly
        parent = '3_0'
        daughters = ['2_1', '1_1']

        assert G.nodes[parent]['division'] is True
        for d in daughters:
            assert G.has_edge(parent, d)
