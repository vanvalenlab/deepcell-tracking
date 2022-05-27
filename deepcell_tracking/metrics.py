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

from collections import Counter

import numpy as np

from deepcell_tracking.trk_io import load_trks
from deepcell_tracking.utils import match_nodes, trk_to_graph


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
        'correct_division': correct,
        'mismatch_division': incorrect,
        'false_positive_division': false_positive,
        'false_negative_division': missed,
        'total_divisions': len(div_gt)
    }


def calculate_summary_stats(correct_division,
                            false_positive_division,
                            false_negative_division,
                            total_divisions,
                            aa_total, aa_tp,
                            te_total, te_tp,
                            n_digits=2):
    """Calculate additional summary statistics for tracking performance
    based on results of classify_divisions

    Catch ZeroDivisionError and set to 0 instead

    Args:
        correct_division (int): True positive or "correct divisions"
        false_positive_division (int): False positives
        false_negative_division (int): False negatives
        total_divisions (int): Total number of ground truth divisions
        aa_total (int): Total number of ground truth associations
        aa_tp (int): True positive associations
        te_total (int): Total number of target assignments
        te_tp (int): True positive target assignments
        n_digits (int, optional): Number of digits to round to. Default 2.
    """

    _round = lambda x: round(x, n_digits)

    try:
        recall = correct_division / (correct_division + false_negative_division)
    except ZeroDivisionError:
        recall = 0

    try:
        precision = correct_division / (correct_division + false_positive_division)
    except ZeroDivisionError:
        precision = 0

    try:
        f1 = 2 * (recall * precision) / (recall + precision)
    except ZeroDivisionError:
        f1 = 0

    try:
        mbc = correct_division / (correct_division + false_negative_division + false_positive_division)
    except ZeroDivisionError:
        mbc = 0

    try:
        fraction_miss = (false_negative_division + false_positive_division) / total_divisions
    except ZeroDivisionError:
        fraction_miss = 0

    try:
        aa = aa_tp / aa_total
    except ZeroDivisionError:
        aa = 0

    try:
        te = te_tp / te_total
    except ZeroDivisionError:
        te = 0

    return {
        'Division Recall': _round(recall),
        'Division Precision': _round(precision),
        'Division F1': _round(f1),
        'Mitotic branching correctness': _round(mbc),
        'Fraction missed divisions': _round(fraction_miss),
        'Association Accuracy': _round(aa),
        'Target Effectiveness': _round(te)
    }


def calculate_association_accuracy(G_gt, G_res):
    """Calculate the association accuracy for each ground truth lineage

    Defined as the number of true positive associations between cells divided by
    the total number of ground truth associations. Associations are equivalent to
    the edges that connect cells in a graph

    Args:
        G_gt (networkx.Graph): Ground truth cell lineage graph.
        G_res (networkx.Graph): Predicted cell lineage graph.

    Returns:
        int: Number of true positive associations
        int: Total number of associations
    """
    true_positive = sum(r in G_gt.edges() for r in G_res.edges())
    total = len(G_gt.edges())


    return true_positive / total


def calculate_target_effectiveness(lineage_gt, lineage_res, cells_gt, cells_res):
    """Calculate the target effectiveness. Final score can be obtained by dividing
    true_positive by total

    The TE measure considers the number of cell instances correctly associated within
    a track with respect to the total number of cells in a track. Only the best possible
    true positive score is recorded for each ground truth lineage

    Args:
        lineage_gt (dict): Ground truth lineages
        linage_res (dict): Predicted lineages
        cells_gt (list): List of ground truthcell ids from `match_nodes`
        cells_res (list): List of result cell ids from `match_nodes`

    Returns:
        int: Number of true positive assignments of cells to lineages
        int: Number of cells present in ground truth
    """
    true_positive = 0
    total = 0

    for g_idx, g_lin in lineage_gt.items():
        # Collect candidates for overlaps, but only save the best
        scores = []

        for r_idx in cells_res[cells_gt == g_idx]:
            r_frames = lineage_res[r_idx]['frames']
            scores.append(sum(r in g_lin['frames'] for r in r_frames))

        # Save total assigments and best score
        total += len(g_lin['frames'])
        if len(scores) > 0:
            true_positive += max(scores)

    return true_positive, total


def benchmark_tracking_performance(trk_gt, trk_res, threshold=1):
    """Compare two related .trk files (one being the GT of the other)

    Calculate division statistics, target effectiveness and association accuracy

    Args:
        trk_gt (path): Path to the ground truth .trk file.
        trk_res (path): Path to the predicted results .trk file.
        threshold (optional, float): threshold value for IoU to count as same cell. Default 1.
            If segmentations are identical, 1 works well.
            For imperfect segmentations try 0.6-0.8 to get better matching
    """
    stats = {}

     # Load data
    trks = load_trks(trk_gt)
    lineage_gt, y_gt = trks['lineages'][0], trks['y']
    trks = load_trks(trk_res)
    lineage_res, y_res = trks['lineages'][0], trks['y']

    # Match up labels in GT to Results to allow for direct comparisons
    cells_gt, cells_res = match_nodes(y_gt, y_res, threshold)

    # Generate graphs
    if len(np.unique(cells_res)) < len(np.unique(cells_gt)):
        node_key = {r: g for g, r in zip(cells_gt, cells_res)}
        # node_key maps gt nodes onto resnodes so must be applied to gt
        G_res = trk_to_graph(lineage_res, node_key=node_key)
        G_gt = trk_to_graph(lineage_gt)
    else:
        node_key = {g: r for g, r in zip(cells_gt, cells_res)}
        G_res = trk_to_graph(lineage_res)
        G_gt = trk_to_graph(lineage_gt, node_key=node_key)

    division_stats = classify_divisions(G_gt, G_res)
    stats.update(division_stats)

    total, tp = calculate_association_accuracy(G_gt, G_res)
    stats['aa_total'] = total
    stats['aa_tp'] = tp

    total, tp = calculate_target_effectiveness(lineage_gt, lineage_res, cells_gt, cells_res)
    stats['te_total'] = total
    stats['te_tp'] = tp

    return stats
