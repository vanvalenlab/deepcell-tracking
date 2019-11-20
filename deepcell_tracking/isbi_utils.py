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


def contig_tracks(label, batch_info, batch_tracked):
    """Check for contiguous tracks (tracks should only consist of consecutive frames).

    Split one track into two if neccesary

    Args:
        label (int): label of the cell.
        batch_info (dict): a track's lineage info
        batch_tracked (dict): the new image data associated with the lineage.

    Returns:
        tuple(dict, dict): updated batch_info and batch_tracked.
    """
    frame_div_missing = False

    frames = batch_info[label]['frames']

    for i, frame in enumerate(frames):
        # If the next frame is available and contiguous we should move on to
        # the next frame. Otherwise, if the next frame is available and
        # NONcontiguous we should separate this track into two.
        if i + 1 <= len(frames) - 1 and frame + 1 != frames[i + 1]:
            frame_div = batch_info[label].get('frame_div')
            if frame_div is None:
                frame_div_missing = True  # TODO: is this necessary?

            # Create a new track to hold the information from this
            # frame forward and add it to the batch.
            new_label = max(batch_info) + 1
            batch_info[new_label] = {
                'old_label': label,
                'label': new_label,
                'frames': frames[i + 1:],
                'daughters': batch_info[label]['daughters'],
                'frame_div': frame_div,
                'parent': None
            }

            for f in frames[i + 1:]:
                batch_tracked[f][batch_tracked[f] == label] = new_label

            # Adjust the info of the current track to vacate the new track info
            batch_info[label]['frames'] = frames[0:i + 1]
            batch_info[label]['daughters'] = []
            batch_info[label]['frame_div'] = None

            break  # Because we are splitting tracks recursively, we stop here

        # If the current frame is the last frame then were done
        # Either the last frame is contiguous and we don't alter batch_info
        # or it's not and it's been made into a new track by the previous
        # iteration of the loop

        if frame_div_missing:
            print('Warning: frame_div is missing')

    return batch_info, batch_tracked


def txt_to_graph(path):
    """Read the ISBI text file and create a Graph.

    Args:
        path (str): Path to the ISBI text file.

    Returns:
        networkx.Graph: Graph representation of the text file.

    Raises:
        ValueError: If the Parent_ID is not in any previous frames.
    """
    names = ['Cell_ID', 'Start', 'End', 'Parent_ID']
    df = pd.read_csv(path, header=None, sep=' ', names=names)

    edges = pd.DataFrame()

    all_ids = set()

    # Add each continuous cell lineage as a set of edges to df
    for _, row in df.iterrows():
        tpoints = np.arange(row['Start'], row['End'] + 1)

        cellids = ['{}_{}'.format(row['Cell_ID'], t) for t in tpoints]

        all_ids.update(cellids)

        edges = edges.append(pd.DataFrame({
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
            continue

        target = '{}_{}'.format(row['Cell_ID'], row['Start'])

        edges = edges.append(pd.DataFrame({
            'source': [source],
            'target': [target]
        }))

        attributes[source] = {'division': True}

    # Create graph
    G = nx.from_pandas_edgelist(edges, source='source', target='target',
                                create_using=nx.DiGraph)
    nx.set_node_attributes(G, attributes)
    return G


def classify_divisions(G_gt, G_res):
    """Compare two graphs and calculate the cell division confusion matrix.

    Args:
        G_gt (networkx.Graph): Ground truth cell lineage graph.
        G_res (networkx.Graph): Predicted cell lineage graph.

    Returns:
        dict: Diciontary of all division statistics.
    """
    # Identify nodes with parent attribute
    div_gt = [node for node, d in G_gt.nodes(data=True)
              if d.get('division', False)]
    div_res = [node for node, d in G_res.nodes(data=True)
               if d.get('division', False)]

    divI = 0   # Correct division
    divJ = 0   # Wrong division
    divC = 0   # False positive division
    divGH = 0  # Missed division

    for node in div_gt:

        pred_gt = list(G_gt.pred[node])
        succ_gt = list(G_gt.succ[node])

        # Check if res node was also called a division
        if node in div_res:
            pred_res = list(G_gt.pred[node])
            succ_res = list(G_res.succ[node])

            # Parents and daughters are the same, perfect!
            if (Counter(pred_gt) == Counter(pred_res) and
                    Counter(succ_gt) == Counter(succ_res)):
                divI += 1

            else:
                if Counter(succ_gt) != Counter(succ_res):
                    print('daughters mismatch, out degree',
                          G_res.out_degree(node))
                if Counter(pred_gt) != Counter(pred_res):
                    print('parents mismatch, in degree',
                          G_res.in_degree(node))
                if G_res.out_degree(node) == G_gt.out_degree(node):
                    print('parent and daughter mismatch, but degree equal at',
                          G_res.out_degree(node))
                divJ += 1

            div_res.remove(node)

        # If not called division, then missed division
        else:
            print('missed division completely')
            divGH += 1

    # Count any remaining res nodes as false positives
    divC += len(div_res)

    return {
        'Correct division': divI,
        'Incorrect division': divJ,
        'False positive division': divC,
        'False negative division': divGH
    }
