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
