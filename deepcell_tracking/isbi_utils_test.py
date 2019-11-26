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
"""Tests for utility functions for the ISBI data format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tempfile

import networkx as nx
import numpy as np

from deepcell_tracking import isbi_utils


class TestIsbiUtils(object):

    def test_trk_to_isbi(self):
        # start with dummy lineage
        # convert to ISBI file
        # read file and validate

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
        with tempfile.NamedTemporaryFile() as temp:
            isbi_utils.trk_to_isbi(track, temp.name)
            data = set(l.decode() for l in temp.readlines())
            expected = {
                '1 0 4 0\n',
                '2 5 5 1\n',
                '3 5 5 1\n',
                '4 7 7 0\n',  # no parent, as it is not consecutive frame
            }
            print(data)
            assert data == expected

    def test_txt_to_graph(self):
        # cell_id, start, end, parent_id
        rows = [
            (1, 0, 3, 0),  # cell 1 is in all 3 frames
            (2, 0, 2, 0),  # cell 2 is not in the last frame
            (3, 3, 3, 2),  # cell 3 is a daughter of 2
            (4, 3, 3, 2),  # cell 4 is a daughter of 2
            (5, 3, 3, 4),  # cell 5 is a daughter of 4, ignored bad frame value
        ]
        with tempfile.NamedTemporaryFile() as text_file:
            # write the file
            for row in rows:
                line = '{} {} {} {}\n'.format(row[0], row[1], row[2], row[3])
                text_file.write(line.encode())

            text_file.flush()  # save the file

            # read the file
            G = isbi_utils.txt_to_graph(text_file.name)
            print(list(G.nodes()))
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
