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
"""Utility functions for the ISBI data format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd


def create_new_ISBI_track(batch_tracked, batch_info, old_label,
                          frames, daughters, frame_div):
    """Adds a new track to the lineage and swaps the labels accordingly.

    Args:
        batch_tracked (dict): tracked data.
        batch_info (dict): tracked info data.
        old_label (int): integer label of the tracked cell.
        frames (list): List of frame numbers in which the cell is present.
        daughters (list): List of daughter cell IDs.
        frame_div (int): Frame number in which the cell divides.

    Returns:
        tuple(dict, dict): updated batch_info and batch_tracked.
    """
    new_label = max(batch_info) + 1

    new_track_data = {
        'old_label': old_label,
        'label': new_label,
        'frames': frames,
        'daughters': daughters,
        'frame_div': frame_div,
        'parent': None
    }

    batch_info[new_label] = new_track_data

    for frame in frames:
        batch_tracked[frame][batch_tracked[frame] == old_label] = new_label

    return batch_info, batch_tracked


def txt_to_graph(path, node_key=None):
    """Read the ISBI text file and create a Graph.

    Args:
        path (str): Path to the ISBI text file.
        node_key (str, optional): Key to identify the parent/daughter links.
            (defaults to Cell_ID+Parent_ID)

    Returns:
        networkx.Graph: Graph representation of the text file.
    """
    names = ['Cell_ID', 'Start', 'End', 'Parent_ID']
    df = pd.read_csv(path, header=None, sep=' ', names=names)

    if node_key is not None:
        df[['Cell_ID', 'Parent_ID']] = df[['Cell_ID', 'Parent_ID']].replace(
            node_key)

    edges = pd.DataFrame()

    # Add each cell lineage as a set of edges to df
    for _, row in df.iterrows():
        tpoints = np.arange(row['Start'], row['End'] + 1)

        cellids = ['{cellid}_{frame}'.format(cellid=row['Cell_ID'], frame=t)
                   for t in tpoints]

        source = cellids[0:-1]
        target = cellids[1:]

        edges = edges.append(pd.DataFrame({
            'source': source,
            'target': target
        }))

    attributes = {}

    # Add parent-daughter connections
    for _, row in df[df['Parent_ID'] != 0].iterrows():
        source = '{cellid}_{frame}'.format(
            cellid=row['Parent_ID'],
            frame=row['Start'] - 1)

        target = '{cellid}_{frame}'.format(
            cellid=row['Cell_ID'],
            frame=row['Start'])

        edges = edges.append(pd.DataFrame({
            'source': [source],
            'target': [target]
        }))

        attributes[source] = {'division': True}

    # Create graph
    G = nx.from_pandas_edgelist(edges, source='source', target='target')
    nx.set_node_attributes(G, attributes)
    return G


def classify_divisions(g_true, g_pred):
    """Compare two graphs and calculate the cell division confusion matrix.

    Args:
        g_true (networkx.Graph): Ground truth cell lineage graph.
        g_pred (networkx.Graph): Predicted cell lineage graph.

    Returns:
        dict: Diciontary of all division statistics.
    """
    div_gt = [node for node, d in g_true.nodes(data='division') if d]
    div_res = [node for node, d in g_pred.nodes(data='division') if d]

    divI = 0   # Correct division
    divJ = 0   # Wrong division
    divC = 0   # False positive division
    divGH = 0  # Missed division

    for node in div_gt:
        nb_gt = list(g_true.neighbors(node))
        # Check if res node was also called a division
        if node in div_res:
            nb_pred = list(g_pred.neighbors(node))
            # If neighbors are same, then correct division
            if Counter(nb_gt) == Counter(nb_pred):
                divI += 1
            # Wrong division
            elif len(nb_pred) == 3:
                divJ += 1
            else:
                divGH += 1
        # If not called division, then missed division
        else:
            divGH += 1

        # Remove processed nodes from pred list
        try:
            div_res.remove(node)
        except ValueError:  # TODO: why did this fail?
            print('attempted removal of node {} failed'.format(node))

    # Count any remaining res nodes as false positives
    divC += len(div_res)

    return {
        'Correct division': divI,
        'Incorrect division': divJ,
        'False positive division': divC,
        'False negative division': divGH
    }
