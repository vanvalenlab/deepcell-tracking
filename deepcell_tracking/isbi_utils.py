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
"""Utility functions for the ISBI data format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from skimage.measure import regionprops

import networkx as nx
import numpy as np
import pandas as pd
import warnings

from deepcell_toolbox import compute_overlap
from deepcell_tracking.utils import load_trks


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


def match_nodes(gt, res):
    """Relabel predicted track to match GT track labels.

    Args:
        gt (np arr): label movie (y) from ground truth .trk file.
        res (np arr): label movie (y) from predicted results .trk file
        threshold (int): threshold value for IoU to count as same cell

    Returns:
        gtcells (np arr): Array of overlapping ids in the gt movie.
        rescells (np arr): Array of overlapping ids in the res movie.

    Raises:
        ValueError: If .
    """
    num_frames = gt.shape[0]
    iou = np.zeros((num_frames, np.max(gt) + 1, np.max(res) + 1))

    # TODO: Compute IOUs only when neccesary
    # If bboxs for true and pred do not overlap with each other, the assignment is immediate
    # Otherwise use pixel-wise IOU to determine which cell is which

    # Regionprops expects one frame at a time
    for frame in range(num_frames):
        gt_frame = gt[frame]
        res_frame = res[frame]

        gt_props = regionprops(np.squeeze(gt_frame.astype('int')))
        gt_boxes = [np.array(gt_prop.bbox) for gt_prop in gt_props]
        gt_boxes = np.array(gt_boxes).astype('double')
        gt_box_labels = [int(gt_prop.label) for gt_prop in gt_props]

        res_props = regionprops(np.squeeze(res_frame.astype('int')))
        res_boxes = [np.array(res_prop.bbox) for res_prop in res_props]
        res_boxes = np.array(res_boxes).astype('double')
        res_box_labels = [int(res_prop.label) for res_prop in res_props]

        overlaps = compute_overlap(gt_boxes, res_boxes)    # has the form [gt_bbox, res_bbox]

        # Find the bboxes that have overlap at all (ind_ corresponds to box number - starting at 0)
        ind_gt, ind_res = np.nonzero(overlaps)

        # frame_ious = np.zeros(overlaps.shape)
        for index in range(ind_gt.shape[0]):

            iou_gt_idx = gt_box_labels[ind_gt[index]]
            iou_res_idx = res_box_labels[ind_res[index]]
            intersection = np.logical_and(gt_frame == iou_gt_idx, res_frame == iou_res_idx)
            union = np.logical_or(gt_frame == iou_gt_idx, res_frame == iou_res_idx)
            iou[frame, iou_gt_idx, iou_res_idx] = intersection.sum() / union.sum()

    gtcells, rescells = np.where(np.nansum(iou, axis=0) >= 1)

    return gtcells, rescells


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
            print('skipped parent %s to daughter %s' % (source, row['Cell_ID']))
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

    WARNING: This function will only work if the labels underlying both
    graphs are the same. E.G. the parents only match if the same label
    splits in the same frame - but each movie isn't guaranteed to be labeled
    in the same way (with the same order). Should be used with match_nodes

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


def benchmark_division_performance(trk_gt, trk_res, path_gt=None, path_res=None):
    """Compare two related .trk files (one being the GT of the other) and measure
    performance on the the divisions in the GT file.

    Args:
        trk_gt (path): Path to the ground truth .trk file.
        trk_res (path): Path to the predicted results .trk file.
        path_gt (path): Desired destination path for the GT ISBI-style .txt
            file (deprecated).
        path_res (path): Desired destination path for the result ISBI-style
            .txt file (deprecated).

    Returns:
        dict: Dictionary of all division statistics.
    """
    # Identify nodes with parent attribute
    # Load both .trk
    trks = load_trks(trk_gt)
    lineage_gt, _, y_gt = trks['lineages'][0], trks['X'], trks['y']
    trks = load_trks(trk_res)
    lineage_res, _, y_res = trks['lineages'][0], trks['X'], trks['y']

    # Produce ISBI style array to work with
    if path_gt is not None or path_res is not None:
        warnings.warn('The `path_gt` and `path_res` arguments are deprecated.',
                      DeprecationWarning)
    gt = trk_to_isbi(lineage_gt, path_gt)
    res = trk_to_isbi(lineage_res, path_res)

    # Match up labels in GT to Results to allow for direct comparisons
    cells_gt, cells_res = match_nodes(y_gt, y_res)

    if len(np.unique(cells_res)) < len(np.unique(cells_gt)):
        node_key = {r: g for g, r in zip(cells_gt, cells_res)}
        # node_key maps gt nodes onto resnodes so must be applied to gt
        G_res = isbi_to_graph(res, node_key=node_key)
        G_gt = isbi_to_graph(gt)
        div_results = classify_divisions(G_gt, G_res)
    else:
        node_key = {g: r for g, r in zip(cells_gt, cells_res)}
        G_res = isbi_to_graph(res)
        G_gt = isbi_to_graph(gt, node_key=node_key)
        div_results = classify_divisions(G_gt, G_res)

    return div_results
