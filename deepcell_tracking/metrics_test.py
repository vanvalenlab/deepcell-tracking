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
"""Tests for utility functions for the ISBI data format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import tarfile
import tempfile
import json
import pytest

import networkx as nx
import numpy as np
import pandas as pd

from deepcell_tracking import metrics
from deepcell_tracking.test_utils import generate_division_data, get_annotated_movie


def test_classify_divisions():
    # Prep graphs
    G = nx.DiGraph()
    G.add_edge('1_0', '1_1')
    G.add_edge('1_1', '1_2')
    G.add_edge('1_2', '1_3')

    G.add_edge('2_0', '2_1')
    G.add_edge('2_1', '2_2')

    # node 2 divides into 3 and 4 in frame 3
    G.add_edge('2_2', '3_3')
    G.add_edge('2_2', '4_3')
    G.nodes['2_2']['division'] = True

    G.add_edge('4_3', '4_4')  # another division in frame 4
    G.nodes['4_3']['division'] = True

    H = G.copy()

    H.nodes['1_3']['division'] = True  # False Positive
    H.nodes['4_3']['division'] = False  # False Negative

    # force an incorrect division
    G.add_edge('3_3', '5_4')  # another division in frame 4
    G.nodes['3_3']['division'] = True
    H.nodes['3_3']['division'] = True

    stats = metrics.classify_divisions(G, H)

    assert len(stats['correct_division']) == 1  # the only correct one
    assert len(stats['false_positive_division']) == 1  # node 1_3
    assert len(stats['false_negative_division']) == 1  # node 4_3
    assert len(stats['mismatch_division']) == 1  # node 3_3
    assert stats['total_divisions'] == 3

    # lists must be the same length
    with pytest.raises(ValueError):
        metrics.classify_divisions(G, H, [], [0])

    # Test with a simple node mapping
    cells_gt, cells_res = np.array([2]), np.array([12])
    node_key = {'2_0': '12_0', '2_1': '12_1', '2_2': '12_2'}
    H_renamed = nx.relabel_nodes(H, node_key)

    stats = metrics.classify_divisions(G, H_renamed, cells_gt=cells_gt, cells_res=cells_res)
    assert len(stats['correct_division']) == 1  # the only correct one
    assert len(stats['false_positive_division']) == 1  # node 1_3
    assert len(stats['false_negative_division']) == 1  # node 4_3
    assert len(stats['mismatch_division']) == 1  # node 3_3
    assert stats['total_divisions'] == 3


def test_correct_shifted_divisions():
    # Generate test data
    y_early, y_late, G_early, G_late = generate_division_data()

    # Generate initial division stats with early as GT
    stats = metrics.classify_divisions(G_early, G_late)
    assert len(stats['false_positive_division']) == 1
    assert len(stats['false_negative_division']) == 1

    # Apply correction
    corrected = metrics.correct_shifted_divisions(
        stats['false_negative_division'],
        stats['false_positive_division'],
        stats['correct_division'],
        y_early, y_late,
        G_early, G_late,
        0.6)

    # Check corrections
    assert len(corrected['correct_division']) == 1
    assert len(stats['false_positive_division']) == 0
    assert len(stats['false_negative_division']) == 0

    # Generate initial division stats with late as GT
    stats = metrics.classify_divisions(G_late, G_early)
    assert len(stats['false_positive_division']) == 1
    assert len(stats['false_negative_division']) == 1

    # Apply correction
    corrected = metrics.correct_shifted_divisions(
        stats['false_negative_division'],
        stats['false_positive_division'],
        stats['correct_division'],
        y_late, y_early,
        G_late, G_early,
        0.6)

    # Check corrections
    assert len(corrected['correct_division']) == 1
    assert len(stats['false_positive_division']) == 0
    assert len(stats['false_negative_division']) == 0


def test_calculate_association_accuracy():
    # lists must be the same length
    with pytest.raises(ValueError):
        metrics.calculate_association_accuracy({}, {}, [], [0])

    tracks_gt = {1: {'label': 1, 'frames': [1, 2, 3], 'daughters': [],
                     'capped': False, 'frame_div': None, 'parent': 3},
                 2: {'label': 2, 'frames': [1, 2], 'daughters': [],
                     'capped': False, 'frame_div': None, 'parent': 3},
                 3: {'label': 3, 'frames': [0], 'daughters': [1, 2],
                     'capped': False, 'frame_div': 1, 'parent': None}}

    tracks_res = copy.deepcopy(tracks_gt)
    tracks_res[1]['frames'] = [1, 2]  # Introduce a missing edge

    # Test with no mapping needed
    tp, total = metrics.calculate_association_accuracy(tracks_gt, tracks_res)
    assert tp == 2
    assert total == 3  # Total edges in gt excluding division connections

    # Map gt 2 onto res 12
    tracks_res[12] = tracks_res[2]
    del tracks_res[2]
    tracks_res[12]['label'] = 12
    tracks_res[3]['daughters'] = [1, 12]

    # Test with mapping of some cells
    tp, total = metrics.calculate_association_accuracy(
        tracks_gt, tracks_res,
        np.array([2]), np.array([12]))
    assert tp == 2
    assert total == 3  # Total edges in gt excluding division connections


def test_calculate_target_effectiveness():
    # lists must be the same length
    with pytest.raises(ValueError):
        metrics.calculate_target_effectiveness({}, {}, [], [0])

    tracks_gt = {1: {'label': 1, 'frames': [1, 2, 3], 'daughters': [],
                     'capped': False, 'frame_div': None, 'parent': 3},
                 2: {'label': 2, 'frames': [1, 2], 'daughters': [],
                     'capped': False, 'frame_div': None, 'parent': 3},
                 3: {'label': 3, 'frames': [0], 'daughters': [1, 2],
                     'capped': False, 'frame_div': 1, 'parent': None}}

    tracks_res = copy.deepcopy(tracks_gt)
    tracks_res[1]['frames'] = [1, 2]  # Introduce a missing edge

    # Test with no mapping needed
    tp, total = metrics.calculate_target_effectiveness(tracks_gt, tracks_res)
    assert tp == 5
    assert total == 6

    # Map gt 2 onto res 12
    tracks_res[12] = tracks_res[2]
    del tracks_res[2]
    tracks_res[12]['label'] = 12
    tracks_res[3]['daughters'] = [1, 12]

    # Test with mapping of some cells
    tp, total = metrics.calculate_target_effectiveness(
        tracks_gt, tracks_res,
        np.array([2]), np.array([12]))
    assert tp == 5
    assert total == 6


def test_benchmark_tracking_performance(tmpdir):
    trk_gt = os.path.join(str(tmpdir), 'test_benchmark_gt.trk')
    trk_res = os.path.join(str(tmpdir), 'test_benchmark_res.trk')

    # Generate lineage data
    tracks_gt = {1: {'label': 1, 'frames': [1, 2], 'daughters': [],
                     'capped': False, 'frame_div': None, 'parent': 3},
                 2: {'label': 2, 'frames': [1, 2], 'daughters': [],
                     'capped': False, 'frame_div': None, 'parent': 3},
                 3: {'label': 3, 'frames': [0], 'daughters': [1, 2],
                     'capped': False, 'frame_div': 1, 'parent': None}}
    X_gt = []
    # Generate tracked movie
    y_gt = get_annotated_movie(img_size=256,
                               labels_per_frame=3,
                               frames=3,
                               mov_type='sequential', seed=0,
                               data_format='channels_last')
    # Let results be same as ground truth
    tracks_res = tracks_gt
    X_res = []
    y_res = y_gt

    # Save gt and res data to .trk files
    with tarfile.open(trk_gt, 'w:gz') as trks:
        # disable auto deletion and close/delete manually
        # to resolve double-opening issue on Windows.
        with tempfile.NamedTemporaryFile('w', delete=False) as lineage:
            json.dump(tracks_gt, lineage, indent=4)
            lineage.flush()
            lineage.close()
            trks.add(lineage.name, 'lineage.json')
            os.remove(lineage.name)

        with tempfile.NamedTemporaryFile(delete=False) as raw:
            np.save(raw, X_gt)
            raw.flush()
            raw.close()
            trks.add(raw.name, 'raw.npy')
            os.remove(raw.name)

        with tempfile.NamedTemporaryFile(delete=False) as tracked:
            np.save(tracked, y_gt)
            tracked.flush()
            tracked.close()
            trks.add(tracked.name, 'tracked.npy')
            os.remove(tracked.name)

    with tarfile.open(trk_res, 'w:gz') as trks:
        # disable auto deletion and close/delete manually
        # to resolve double-opening issue on Windows.
        with tempfile.NamedTemporaryFile('w', delete=False) as lineage:
            json.dump(tracks_res, lineage, indent=4)
            lineage.flush()
            lineage.close()
            trks.add(lineage.name, 'lineage.json')
            os.remove(lineage.name)

        with tempfile.NamedTemporaryFile(delete=False) as raw:
            np.save(raw, X_res)
            raw.flush()
            raw.close()
            trks.add(raw.name, 'raw.npy')
            os.remove(raw.name)

        with tempfile.NamedTemporaryFile(delete=False) as tracked:
            np.save(tracked, y_res)
            tracked.flush()
            tracked.close()
            trks.add(tracked.name, 'tracked.npy')
            os.remove(tracked.name)

    stats = metrics.benchmark_tracking_performance(trk_gt, trk_res)
    assert stats['correct_division'] == 1
    assert stats['mismatch_division'] == 0
    assert stats['false_positive_division'] == 0
    assert stats['false_negative_division'] == 0
    assert stats['total_divisions'] == 1

    for k in ['aa_total', 'aa_tp', 'te_total', 'te_tp']:
        assert k in stats
