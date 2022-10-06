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
"""Utility functions for the ISBI data format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import networkx as nx
import numpy as np
import pandas as pd

# Imports for backwards compatibility
from deepcell_tracking.utils import match_nodes, contig_tracks
from deepcell_tracking.metrics import calculate_summary_stats
from deepcell_tracking.metrics import benchmark_tracking_performance


def benchmark_division_performance(trk_gt, trk_res):
    warnings.warn('benchmark_division_performance is deprecated. '
                  'Please use deepcell_tracking.metrics.benchmark_tracking_performance instead',
                  DeprecationWarning)

    return benchmark_tracking_performance(trk_gt, trk_res)


def trk_to_isbi(track, path=None):
    """Convert a lineage track into an ISBI formatted text file.

    Args:
        track (dict): Cell lineage object.
        path (str): Path to save the .txt file (deprecated).

    Returns:
        pd.DataFrame: DataFrame of ISBI data for each label.
    """
    isbi = []
    for label in track:
        first_frame = min(track[label]['frames'])
        last_frame = max(track[label]['frames'])
        parent = track[label]['parent']
        parent = 0 if parent is None else parent
        if parent:
            parent_frames = track[parent]['frames']
            if parent_frames[-1] != first_frame - 1:
                parent = 0

        isbi_dict = {'Cell_ID': label,
                     'Start': first_frame,
                     'End': last_frame,
                     'Parent_ID': parent}
        isbi.append(isbi_dict)

    if path is not None:
        with open(path, 'w') as text_file:
            for cell in isbi_dict:
                line = '{cell_id} {start} {end} {parent}\n'.format(
                    cell_id=cell['Cell_ID'],
                    start=cell['Start'],
                    end=cell['End'],
                    parent=cell['Parent_ID']
                )
                text_file.write(line)
    df = pd.DataFrame(isbi)
    return df


def txt_to_graph(path, node_key=None):
    """Read the ISBI text file and create a Graph.

    Args:
        path (str): Path to the ISBI text file.
        node_key (dict): Map between gt nodes and result nodes

    Returns:
        networkx.Graph: Graph representation of the ISBI data.

    Raises:
        ValueError: If the Parent_ID is not in any previous frames.
    """
    names = ['Cell_ID', 'Start', 'End', 'Parent_ID']
    df = pd.read_csv(path, header=None, sep=' ', names=names)
    G = isbi_to_graph(df, node_key)
    return G


def isbi_to_graph(df, node_key=None):
    """Create a Graph from DataFrame of ISBI info.

    Args:
        data (pd.DataFrame): DataFrame of ISBI-style info.
        node_key (dict): Map between gt nodes and result nodes

    Returns:
        networkx.Graph: Graph representation of the ISBI data.

    Raises:
        ValueError: If the Parent_ID is not in any previous frames.
    """
    if node_key is not None:
        df[['Cell_ID', 'Parent_ID']] = df[['Cell_ID', 'Parent_ID']].replace(node_key)

    edges = []

    all_ids = set()
    single_nodes = set()

    # Add each continuous cell lineage as a set of edges to df
    for _, row in df.iterrows():
        tpoints = np.arange(row['Start'], row['End'] + 1)

        cellids = ['{}_{}'.format(row['Cell_ID'], t) for t in tpoints]

        if len(cellids) == 1:
            single_nodes.add(cellids[0])

        all_ids.update(cellids)

        edges.append(pd.DataFrame({
            'source': cellids[0:-1],
            'target': cellids[1:],
        }))

    attributes = {}

    # Add parent-daughter connections
    for _, row in df[df['Parent_ID'] != 0].iterrows():
        # Assume the parent is in the previous frame.
        parent_frame = row['Start'] - 1
        source = '{}_{}'.format(row['Parent_ID'], parent_frame)

        if source not in all_ids:  # parents should be in the previous frame.
            # parent_frame = df[df['Cell_ID'] == row['Parent_id']]['End']
            # source = '{}_{}'.format(row['Parent_ID'], parent_frame)
            print('skipped parent %s to daughter %s' % (source, row['Cell_ID']))
            continue

        target = '{}_{}'.format(row['Cell_ID'], row['Start'])

        edges.append(pd.DataFrame({
            'source': [source],
            'target': [target]
        }))

        attributes[source] = {'division': True}

    # Create graph
    edges = pd.concat(edges)
    G = nx.from_pandas_edgelist(edges, source='source', target='target',
                                create_using=nx.DiGraph)
    nx.set_node_attributes(G, attributes)

    # Add all isolates to graph
    for cell_id in single_nodes:
        G.add_node(cell_id)
    return G
