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
"""Tests for utility functions for the ISBI data format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import tarfile
import tempfile
import json

import networkx as nx
import numpy as np
import pandas as pd

from deepcell_tracking import isbi_utils
from deepcell_tracking.test_utils import get_annotated_movie


class TestIsbiUtils(object):

    def test_trk_to_isbi(self, tmpdir):
        # start with dummy lineage
        # convert to ISBI array
        # validate array

        track = {}
        # first cell, skips frame 3 but divides in frame 4
        track[1] = {
            'frames': [0, 1, 2, 4],  # skipped a frame
            'daughters': [2, 3],
            'frame_div': 4,
            'parent': None,
            'label': 1,
        }
        track[2] = {
            'frames': [5],
            'daughters': [],
            'frame_div': None,
            'parent': 1,
            'label': 2,
        }
        track[3] = {
            'frames': [5],
            'daughters': [4],  # parent not in previous frame
            'frame_div': 5,
            'parent': 1,
            'label': 3,
        }
        track[4] = {
            'frames': [7],
            'daughters': [],
            'frame_div': None,
            'parent': 3,
            'label': 4,
        }
        df = isbi_utils.trk_to_isbi(track)

        expected = [{'Cell_ID': 1, 'Start': 0, 'End': 4, 'Parent_ID': 0},
                    {'Cell_ID': 2, 'Start': 5, 'End': 5, 'Parent_ID': 1},
                    {'Cell_ID': 3, 'Start': 5, 'End': 5, 'Parent_ID': 1},
                    {'Cell_ID': 4, 'Start': 7, 'End': 7, 'Parent_ID': 0}]
        expected_df = pd.DataFrame(expected)
        assert df.equals(expected_df)

    def test_txt_to_graph(self, tmpdir):
        # cell_id, start, end, parent_id
        rows = [
            (1, 0, 3, 0),  # cell 1 is in all 3 frames
            (2, 0, 2, 0),  # cell 2 is not in the last frame
            (3, 3, 3, 2),  # cell 3 is a daughter of 2
            (4, 3, 3, 2),  # cell 4 is a daughter of 2
            (5, 3, 3, 4),  # cell 5 is a daughter of 4, ignored bad frame value
        ]
        text_file = os.path.join(str(tmpdir), 'test_txt_to_graph.txt')
        with open(text_file, 'wb') as f:
            # write the file
            for row in rows:
                line = '{} {} {} {}{}'.format(
                    row[0], row[1], row[2], row[3], os.linesep)
                f.write(line.encode())

            f.flush()  # save the file

        # read the file
        G = isbi_utils.txt_to_graph(text_file)
        for row in rows:
            node_ids = ['{}_{}'.format(row[0], t)
                        for t in range(row[1], row[2] + 1)]

            for node_id in node_ids:
                assert node_id in G

            if row[3]:  # should have a division
                daughter_id = '{}_{}'.format(row[0], row[1])
                parent_id = '{}_{}'.format(row[3], row[1] - 1)
                if G.has_node(parent_id):
                    assert G.nodes[parent_id]['division'] is True
                    assert G.has_edge(parent_id, daughter_id)
                else:
                    assert not G.in_degree(daughter_id)

    def test_isbi_to_graph(self):
        # cell_id, start, end, parent_id
        data = [{'Cell_ID': 1, 'Start': 0, 'End': 3, 'Parent_ID': 0},
                {'Cell_ID': 2, 'Start': 0, 'End': 2, 'Parent_ID': 0},
                {'Cell_ID': 3, 'Start': 3, 'End': 3, 'Parent_ID': 2},
                {'Cell_ID': 4, 'Start': 3, 'End': 3, 'Parent_ID': 2},
                {'Cell_ID': 5, 'Start': 3, 'End': 3, 'Parent_ID': 4}]
        df = pd.DataFrame(data)
        G = isbi_utils.isbi_to_graph(df)
        for d in data:
            node_ids = ['{}_{}'.format(d["Cell_ID"], t)
                        for t in range(d["Start"], d["End"] + 1)]

            for node_id in node_ids:
                assert node_id in G

            if d["Parent_ID"]:  # should have a division
                daughter_id = '{}_{}'.format(d["Cell_ID"], d["Start"])
                parent_id = '{}_{}'.format(d["Parent_ID"], d["Start"] - 1)
                if G.has_node(parent_id):
                    assert G.nodes[parent_id]['division'] is True
                    assert G.has_edge(parent_id, daughter_id)
                else:
                    assert not G.in_degree(daughter_id)

    def test_classify_divisions(self):
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

        stats = isbi_utils.classify_divisions(G, H)

        assert stats['Correct division'] == 1  # the only correct one
        assert stats['False positive division'] == 1  # node 1_3
        assert stats['False negative division'] == 1  # node 4_3
        assert stats['Incorrect division'] == 1  # node 3_3

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
        new_track, _ = isbi_utils.contig_tracks(1, track, y)
        assert original_track == new_track

        # test non-contiguous
        track = copy.copy(original_track)
        track[1]['frames'].append(4)
        new_track, _ = isbi_utils.contig_tracks(1, track, y)

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
        gtcells, rescells = isbi_utils.match_nodes(y1, y1)
        for gt_cell, res_cell in zip(gtcells, rescells):
            assert gt_cell == res_cell

        # test different movie (with known values)
        y2 = get_annotated_movie(img_size=256,
                                 labels_per_frame=labels_per_frame,
                                 frames=frames,
                                 mov_type='sequential', seed=1,
                                 data_format='channels_last')
        gtcells, rescells = isbi_utils.match_nodes(y1, y2)

        assert len(rescells) == len(gtcells)
        for loc, gt_cell in enumerate(np.unique(gtcells)):
            # because movies have the same first frame, every
            # iteration of unique values should match original label
            assert gt_cell == rescells[loc * 3]

    def test_benchmark_division_performance(self, tmpdir):
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

        expected = {'Correct division': 1, 'Incorrect division': 0,
                    'False positive division': 0, 'False negative division': 0}
        results = isbi_utils.benchmark_division_performance(trk_gt, trk_res)
        assert results == expected
