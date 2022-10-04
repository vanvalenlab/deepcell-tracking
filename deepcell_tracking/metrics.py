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
"""Functions for evaluating tracking performance"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import itertools

import numpy as np

from deepcell_tracking.trk_io import load_trks
from deepcell_tracking.utils import match_nodes, trk_to_graph


def map_node(gt_node, G_res, cells_gt, cells_res):
    """Finds the res node that matches the gt_node submitted

    Args:
        gt_node (str): String matching form '{cell id}_{frame}'
        G_res (networkx.graph): Graph of the results
        cells_gt (np.array): Array containing ground truth cell ids corresponding to res ids
        cells_res (np.array): Array containing corresponding res ids
    """
    idx = int(gt_node.split('_')[0])
    frame = int(gt_node.split('_')[1])

    if idx in cells_gt:
        for r_idx in cells_res[cells_gt == idx]:
            # Check if node exists with the right frame
            r_node = '{}_{}'.format(r_idx, frame)
            if r_node in G_res.nodes:
                return r_node
        else:
            # Can't find result node so return original gt node
            return gt_node
    elif gt_node in G_res.nodes:
        return gt_node
    else:
        return gt_node


def calculate_association_accuracy(lineage_gt, lineage_res, cells_gt=[], cells_res=[]):
    """Calculate the association accuracy for each ground truth lineage

    Defined as the number of true positive associations between cells divided by
    the total number of ground truth associations. Associations are equivalent to
    the edges that connect cells in a graph

    Args:
        lineage_gt (dict): Ground truth lineages
        linage_res (dict): Predicted lineages
        cells_gt (list): List of ground truth cell ids from `match_nodes`
        cells_res (list): List of result cell ids from `match_nodes`

    Returns:
        int: Number of true positive associations
        int: Total number of associations

    Raises:
        ValueError: cells_gt and cells_res must be the same length
    """
    if len(cells_gt) != len(cells_res):
        raise ValueError('cells_gt and cells_res must be the same length.')

    true_positive = 0
    total = 0

    for g_idx, g_lin in lineage_gt.items():
        # Calculate gt edges
        g_frames = g_lin['frames']
        g_edges = ['{}-{}'.format(t0, t1) for t0, t1 in zip(g_frames[:-1], g_frames[1:])]
        total += len(g_edges)

        # Check for any mappings
        if g_idx in cells_gt:
            scores = []
            for r_idx in cells_res[cells_gt == g_idx]:
                r_frames = lineage_res[r_idx]['frames']
                r_edges = ['{}-{}'.format(t0, t1) for t0, t1 in zip(r_frames[:-1], r_frames[1:])]
                scores.append(sum(r in g_edges for r in r_edges))
            true_positive += max(scores)

        # Check if the idx already matches
        elif g_idx in lineage_res:
            r_frames = lineage_res[g_idx]['frames']
            r_edges = ['{}-{}'.format(t0, t1) for t0, t1 in zip(r_frames[:-1], r_frames[1:])]
            true_positive += sum(r in g_edges for r in r_edges)

    return true_positive, total


def calculate_target_effectiveness(lineage_gt, lineage_res, cells_gt=[], cells_res=[]):
    """Calculate the target effectiveness. Final score can be obtained by dividing
    true_positive by total

    The TE measure considers the number of cell instances correctly associated within
    a track with respect to the total number of cells in a track. Only the best possible
    true positive score is recorded for each ground truth lineage

    Args:
        lineage_gt (dict): Ground truth lineages
        linage_res (dict): Predicted lineages
        cells_gt (list): List of ground truth cell ids from `match_nodes`
        cells_res (list): List of result cell ids from `match_nodes`

    Returns:
        int: Number of true positive assignments of cells to lineages
        int: Number of cells present in ground truth

    Raises:
        ValueError: cells_gt and cells_res must be the same length
    """
    if len(cells_gt) != len(cells_res):
        raise ValueError('cells_gt and cells_res must be the same length.')

    true_positive = 0
    total = 0

    for g_idx, g_lin in lineage_gt.items():
        # Check for any mappings
        if g_idx in cells_gt:
            # Collect candidates for overlaps, but only save the best
            scores = []
            for r_idx in cells_res[cells_gt == g_idx]:
                r_frames = lineage_res[r_idx]['frames']
                scores.append(sum(r in g_lin['frames'] for r in r_frames))

            true_positive += max(scores)

        # Check if the idx already matches
        elif g_idx in lineage_res:
            r_frames = lineage_res[g_idx]['frames']
            true_positive += sum(r in g_lin['frames'] for r in r_frames)

        # Save total assigments for this gt lineage
        total += len(g_lin['frames'])

    return true_positive, total


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
        mbc = correct_division / (correct_division
                                  + false_negative_division
                                  + false_positive_division)
    except ZeroDivisionError:
        mbc = 0

    try:
        fraction_miss = false_negative_division / total_divisions
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


class TrackingMetrics:
    def __init__(self, lineage_gt, y_gt, lineage_res, y_res, threshold=1, allow_division_shift=True):
        """Class to coordinate the benchmarking of a pair of trk files

        Args:
            trk_gt (path): Path to the ground truth .trk file.
            trk_res (path): Path to the predicted results .trk file.
            threshold (optional, float): threshold value for IoU to count as same cell. Default 1.
                If segmentations are identical, 1 works well.
                For imperfect segmentations try 0.6-0.8 to get better matching
            allow_temporal_shifts (optional, bool): Allows divisions to be treated as correct if
                they are off by a single frame. Default True.
        """

        self.lineage = {'gt': lineage_gt, 'res': lineage_res}
        self.y = {'gt': y_gt, 'res': y_res}
        self.threshold = threshold

        # Match up labels in GT to Results to allow for direct comparisons
        self.cells_gt, self.cells_res = match_nodes(y_gt, y_res, self.threshold)

        # Generate graphs without remapping nodes to avoid losing lineages
        self.G = {'gt': trk_to_graph(lineage_gt), 'res': trk_to_graph(lineage_res)}

        # Classify divison errors
        self.classify_divisions()

        if allow_division_shift:
            self.correct_shifted_divisions()

    @classmethod
    def from_trk_files(cls, trk_gt, trk_res, threshold=1, allow_division_shift=True):
        # Load data
        trks = load_trks(trk_gt)
        lineage_gt, y_gt = trks['lineages'][0], trks['y']
        trks = load_trks(trk_res)
        lineage_res, y_res = trks['lineages'][0], trks['y']

        return cls(
            lineage_gt=lineage_gt, y_gt=y_gt,
            lineage_res=lineage_res, y_res=y_res,
            threshold=threshold,
            allow_division_shift=allow_division_shift
        )

    def classify_divisions(self):
        """Compare two graphs and calculate the cell division confusion matrix.

        WARNING: This function will only work if the labels underlying both
        graphs are the same. E.G. the parents only match if the same label
        splits in the same frame - but each movie isn't guaranteed to be labeled
        in the same way (with the same order). Should be used with match_nodes

        Returns:
            dict: Diciontary of all division statistics
        """

        def _map_node(gt_node):
            return map_node(gt_node, self.G['res'], self.cells_gt, self.cells_res)

        # Identify nodes with parent attribute
        div_gt = [node for node, d in self.G['gt'].nodes(data=True)
                if d.get('division', False)]
        div_res = [node for node, d in self.G['res'].nodes(data=True)
                if d.get('division', False)]

        # Record total number of gt divisions
        self.total_divisions = len(div_gt)

        self.correct = []         # Correct division
        self.incorrect = []       # Wrong division
        self.missed = []          # Missed division

        for node in div_gt:
            idx = int(node.split('_')[0])
            frame = int(node.split('_')[1])

            # Check if the index is mapped onto a different results index
            if idx in self.cells_gt:
                for r_idx in self.cells_res[self.cells_gt == idx]:
                    # Check if node exists with the right frame
                    r_node = '{}_{}'.format(r_idx, frame)
                    if r_node in self.G['res'].nodes:
                        break  # Exit for loop since we found the right node
                else:
                    # Node doesn't exist so count this division as missed
                    print('missed node {} division completely'.format(node))
                    self.missed.append(node)
                    continue  # move on to next node in div_gt
            # Check if the node exists with same id in G_res
            elif node in self.G['res'].nodes:
                r_node = node
            # Node doesn't exist
            else:
                print('missed node {} division completely'.format(node))
                self.missed.append(node)
                continue  # move on to next node in div_gt

            # If we found the results node, evaluate division result
            # Get gt predecessors and successors for comparsion
            # Map gt nodes onto results nodes if possible
            pred_gt = [_map_node(n) for n in self.G['gt'].pred[node]]
            succ_gt = [_map_node(n) for n in self.G['gt'].succ[node]]

            # Check if res node was also called a division
            if r_node in div_res:
                # Get res predecessors and successor
                pred_res = list(self.G['res'].pred[r_node])
                succ_res = list(self.G['res'].succ[r_node])

                # Parents and daughters are the same, perfect!
                if (Counter(pred_gt) == Counter(pred_res) and
                        Counter(succ_gt) == Counter(succ_res)):
                    self.correct.append(node)

                else:  # what went wrong?
                    self.incorrect.append(node)
                    errors = ['out degree = {}'.format(self.G['res'].out_degree(r_node))]
                    if Counter(succ_gt) != Counter(succ_res):
                        errors.append('daughters mismatch')
                    if Counter(pred_gt) != Counter(pred_res):
                        errors.append('parents mismatch')
                    if self.G['res'].out_degree(r_node) == self.G['gt'].out_degree(node):
                        errors.append('gt and res degree equal')
                    print(node, '{}.'.format(', '.join(errors)))

                div_res.remove(r_node)

            else:  # valid division not in results, it was missed
                print('missed node {} division completely'.format(node))
                self.missed.append(node)

        # Count any remaining res nodes as false positives
        self.false_positive = div_res

    def correct_shifted_divisions(self):
        """Correct divisions errors that are shifted by a frame and should be counted as correct
        """

        # Explicitly label nodes according to source
        missed = ['gt-'+n for n in self.missed]
        false_positive = ['res-'+n for n in self.false_positive]

        # Convert to dictionary for lookup by frame
        d_missed, d_fp = {}, {}
        for d, l in [(d_missed, missed), (d_fp, false_positive)]:
            for n in l:
                t = int(n.split('_')[-1])
                v = d.get(t, [])
                v.append(n)
                d[t] = v

        frame_pairs = []
        for t in d_missed:
            if t+1 in d_fp:
                frame_pairs.append((t, t+1))
            if t-1 in d_fp:
                frame_pairs.append((t-1, t))

        # Convert to set to remove any duplicates
        frame_pairs = list(set(frame_pairs))

        matches = []

        # Loop over each pair of frames
        for t1, t2 in frame_pairs:
            # Get nodes from each frames
            n1s = d_missed.get(t1, []) + d_fp.get(t1, [])
            n2s = d_missed.get(t2, []) + d_fp.get(t2, [])

            # Compare each pair and save if they are above the threshold
            for n1, n2 in itertools.product(n1s, n2s):
                source1, node1 = n1.split('-')[0], n1.split('-')[1]
                source2, node2 = n2.split('-')[0], n2.split('-')[1]

                # Check if the nodes are from different sources
                if source1 == source2:
                    continue

                # Compare sum of daughters in n1 to parent in n2
                daughters = [int(d.split('_')[0]) for d in list(self.G[source1].succ[node1])]
                mask1 = np.logical_or(self.y[source1][t2] == daughters[0], self.y[source1][t2] == daughters[1])
                mask2 = self.y[source2][t2] == int(node2.split('_')[0])

                # Compute iou
                intersection = np.logical_and(mask1, mask2)
                union = np.logical_or(mask1, mask2)
                iou = intersection.sum() / union.sum()
                if iou >= self.threshold:
                    matches.extend([n1, n2])

        # Remove matches from the list of errors
        for n in matches:
            source, node = n.split('-')[0], n.split('-')[1]
            if source == 'gt':
                self.missed.remove(node)
            elif source == 'res':
                self.false_positive.remove(node)

    @property
    def stats(self):
        return {
            'correct_division': self.correct,
            'mismatch_division': self.incorrect,
            'false_positive_division': self.false_positive,
            'false_negative_division': self.missed,
            'total_divisions': self.total_divisions
        }


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

    # Generate graphs without remapping nodes to avoid losing lineages
    G_gt = trk_to_graph(lineage_gt)
    G_res = trk_to_graph(lineage_res)

    # Calculate metrics
    division_stats = classify_divisions(G_gt, G_res, cells_gt, cells_res)
    stats.update(division_stats)

    stats['aa_tp'], stats['aa_total'] = calculate_association_accuracy(lineage_gt, lineage_res,
                                                                       cells_gt, cells_res)

    stats['te_tp'], stats['te_total'] = calculate_target_effectiveness(lineage_gt, lineage_res,
                                                                       cells_gt, cells_res)

    return stats
