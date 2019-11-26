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


def trk_to_isbi(track, path):
    """Convert a lineage track into an ISBI formatted text file.

    Args:
        track (dict): Cell lineage object.
        path (str): Path to save the .txt file.
    """
    with open(path, 'w') as text_file:
        for label in track:
            first_frame = min(track[label]['frames'])
            last_frame = max(track[label]['frames'])
            parent = track[label]['parent']
            parent = 0 if parent is None else parent
            if parent:
                parent_frames = track[parent]['frames']
                if parent_frames[-1] != first_frame - 1:
                    parent = 0

            line = '{cell_id} {start} {end} {parent}\n'.format(
                cell_id=label,
                start=first_frame,
                end=last_frame,
                parent=parent
            )

            text_file.write(line)


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

            for d in batch_info[new_label]['daughters']:
                batch_info[d]['parent'] = new_label

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
    single_nodes = set()

    # Add each continuous cell lineage as a set of edges to df
    for _, row in df.iterrows():
        tpoints = np.arange(row['Start'], row['End'] + 1)

        cellids = ['{}_{}'.format(row['Cell_ID'], t) for t in tpoints]

        if len(cellids) == 1:
            single_nodes.add(cellids[0])

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
            print('%s: skipped parent %s to daughter %s' % (path, source, row['Cell_ID']))
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

    # Add all isolates to graph
    for cell_id in single_nodes:
        G.add_node(cell_id)
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

    correct = 0         # Correct division
    incorrect = 0       # Wrong division
    false_positive = 0  # False positive division
    missed = 0          # Missed division

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
                correct += 1

            else:  # what went wrong?
                incorrect += 1
                errors = ['out degree = {}'.format(G_res.out_degree(node))]
                if Counter(succ_gt) != Counter(succ_res):
                    errors.append('daughters mismatch')
                if Counter(pred_gt) != Counter(pred_res):
                    errors.append('parents mismatch')
                if G_res.out_degree(node) == G_gt.out_degree(node):
                    errors.append('gt and res degree equal')
                print(node, '{}.'.format(', '.join(errors)))

            div_res.remove(node)

        else:  # valid division not in results, it was missed
            print('missed node {} division completely'.format(node))
            missed += 1

    # Count any remaining res nodes as false positives
    false_positive += len(div_res)

    return {
        'Correct division': correct,
        'Incorrect division': incorrect,
        'False positive division': false_positive,
        'False negative division': missed
    }
