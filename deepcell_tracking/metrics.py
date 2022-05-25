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
from skimage.measure import regionprops

import networkx as nx
import numpy as np
import pandas as pd
import warnings

from deepcell_toolbox import compute_overlap
from deepcell_tracking.trk_io import load_trks
from deepcell_tracking.isbi_utils import trk_to_isbi, isbi_to_graph, match_nodes


class Metrics:
    def __init__(self, trk_gt, trk_res, threshold=1):
        """Compare two related .trk files (one being the GT of the other)

        Args:
            trk_gt (path): Path to the ground truth .trk file.
            trk_res (path): Path to the predicted results .trk file.
            threshold (optional, float): threshold value for IoU to count as same cell. Default 1.
                If segmentations are identical, 1 works well.
                For imperfect segmentations try 0.6-0.8 to get better matching
        """

        # Load data
        trks = load_trks(trk_gt)
        self.lineage_gt, self.y_gt = trks['lineages'][0], trks['y']
        trks = load_trks(trk_res)
        self.lineage_res, self.y_res = trks['lineages'][0], trks['y']

        # Produce ISBI style array to work with
        gt = trk_to_isbi(self.lineage_gt)
        res = trk_to_isbi(self.lineage_res)

        # Match up labels in GT to Results to allow for direct comparisons
        self.cells_gt, self.cells_res = match_nodes(self.y_gt, self.y_res, threshold)

        # Generate graphs
        if len(np.unique(self.cells_res)) < len(np.unique(self.cells_gt)):
            node_key = {r: g for g, r in zip(self.cells_gt, self.cells_res)}
            # node_key maps gt nodes onto resnodes so must be applied to gt
            self.G_res = isbi_to_graph(res, node_key=node_key)
            self.G_gt = isbi_to_graph(gt)
        else:
            node_key = {g: r for g, r in zip(self.cells_gt, self.cells_res)}
            self.G_res = isbi_to_graph(res)
            self.G_gt = isbi_to_graph(gt, node_key=node_key)

        # Initialize division metrics
        self.correct = 0         # Correct division
        self.incorrect = 0       # Wrong division
        self.false_positive = 0  # False positive division
        self.missed = 0          # Missed division
        self.total_div = 0       # Total divisions in ground truth

        self._classify_divisions()

    def _classify_divisions(self):
        """Compare two graphs and calculate the cell division confusion matrix.

        WARNING: This function will only work if the labels underlying both
        graphs are the same. E.G. the parents only match if the same label
        splits in the same frame - but each movie isn't guaranteed to be labeled
        in the same way (with the same order). Should be used with match_nodes
        """
        # Identify nodes with parent attribute
        div_gt = [node for node, d in self.G_gt.nodes(data=True)
                  if d.get('division', False)]
        div_res = [node for node, d in self.G_res.nodes(data=True)
                   if d.get('division', False)]

        for node in div_gt:

            pred_gt = list(self.G_gt.pred[node])
            succ_gt = list(self.G_gt.succ[node])

            # Check if res node was also called a division
            if node in div_res:
                pred_res = list(self.G_gt.pred[node])
                succ_res = list(self.G_res.succ[node])

                # Parents and daughters are the same, perfect!
                if (Counter(pred_gt) == Counter(pred_res) and
                        Counter(succ_gt) == Counter(succ_res)):
                    self.correct += 1

                else:  # what went wrong?
                    self.incorrect += 1
                    errors = ['out degree = {}'.format(self.G_res.out_degree(node))]
                    if Counter(succ_gt) != Counter(succ_res):
                        errors.append('daughters mismatch')
                    if Counter(pred_gt) != Counter(pred_res):
                        errors.append('parents mismatch')
                    if self.G_res.out_degree(node) == self.G_gt.out_degree(node):
                        errors.append('gt and res degree equal')
                    print(node, '{}.'.format(', '.join(errors)))

                div_res.remove(node)

            else:  # valid division not in results, it was missed
                print('missed node {} division completely'.format(node))
                self.missed += 1

        # Count any remaining res nodes as false positives
        self.false_positive += len(div_res)

        # Count total ground truth divisions
        self.total_div = len(div_gt)


class DivisionReport:
    def __init__(self):
        # Collect Metrics objects
        self.results = []

        # Initialize division metrics
        self.correct = 0         # Correct division / true positive
        self.incorrect = 0       # Wrong division
        self.false_positive = 0  # False positive division
        self.missed = 0          # Missed division / false negative
        self.total_div = 0       # Total divisions in ground truth

    def add_metrics(self, metrics):
        if not isinstance(metrics, Metrics):
            raise ValueError('Must be an instance of `Metrics`')

        self.correct += metrics.correct
        self.incorrect += metrics.incorrect
        self.false_positive += metrics.false_positive
        self.missed += metrics.missed
        self.total_div += metrics.total_div

    @property
    def recall(self):
        try:
            recall = self.correct / (self.correct + self.missed)
        except ZeroDivisionError:
            recall = 0
        return recall

    @property
    def precision(self):
        try:
            precision = self.correct / (self.correct + self.false_positive)
        except ZeroDivisionError:
            precision = 0
        return precision

    @property
    def f1(self):
        try:
            f1 = 2 * (self.recall * self.precision) / (self.recall + self.precision)
        except ZeroDivisionError:
            f1 = 0
        return f1

    @property
    def mitotic_branching_correctness(self):
        try:
            mbc = self.correct / (self.correct + self.missed + self.false_positive)
        except ZeroDivisionError:
            mbc = 0
        return mbc

    @property
    def fraction_missed(self):
        try:
            frac = (self.missed + self.false_positive) / self.total_div
        except ZeroDivisionError:
            frac = 0
        return frac

