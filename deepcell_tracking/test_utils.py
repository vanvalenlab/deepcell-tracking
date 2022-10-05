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
"""Utilities for testing deepcell-tracking"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
import skimage as sk


def get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


def get_annotated_image(img_size=256, num_labels=3, sequential=True, seed=1):
    np.random.seed(seed)
    num_labels_act = False
    trial = 0
    while num_labels != num_labels_act:
        if trial > 10:
            raise Exception('Labels have merged despite 10 different random seeds.'
                            ' Increase image size or reduce the number of labels')
        im = np.zeros((img_size, img_size))
        points = img_size * np.random.random((2, num_labels))
        im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
        im = sk.filters.gaussian(im, sigma=5)
        blobs = im > 0.7 * im.mean()
        all_labels, num_labels_act = sk.measure.label(blobs, return_num=True)
        if num_labels != num_labels_act:
            seed += 1
            np.random.seed(seed)
            trial += 1

    if not sequential:
        labels_in_frame = np.unique(all_labels)
        for label in range(num_labels):
            curr_label = label + 1
            new_label = np.random.randint(1, num_labels * 100)
            while new_label in labels_in_frame:
                new_label = np.random.randint(1, num_labels * 100)
            labels_in_frame = np.append(labels_in_frame, new_label)
            label_loc = np.where(all_labels == curr_label)
            all_labels[:, :][label_loc] = new_label

    return all_labels.astype('int32')


def get_annotated_movie(img_size=256, labels_per_frame=3, frames=3,
                        mov_type='sequential', seed=1,
                        data_format='channels_last'):
    if mov_type in ('sequential', 'repeated'):
        sequential = True
    elif mov_type == 'random':
        sequential = False
    else:
        raise ValueError('mov_type must be one of "sequential", '
                         '"repeated" or "random"')

    if data_format == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 0

    y = []
    while len(y) < frames:
        _y = get_annotated_image(img_size=img_size, num_labels=labels_per_frame,
                                 sequential=sequential, seed=seed)
        y.append(_y)
        seed += 1

    y = np.stack(y, axis=0)  # expand to 3D

    if mov_type == 'sequential':
        for frame in range(frames):
            if frame == 0:
                new_label = labels_per_frame
                continue
            for label in range(labels_per_frame):
                curr_label = label + 1
                new_label += 1
                label_loc = np.where(y[frame, :, :] == curr_label)
                y[frame, :, :][label_loc] = new_label

    y = np.expand_dims(y, axis=channel_axis)

    return y.astype('int32')


def generate_division_data(img_size=256):
    parent_id = 10

    # Generate parent frame
    im = np.zeros((img_size, img_size))
    parent = np.random.randint(low=img_size * 0.25, high=img_size * 0.75, size=(2,))
    im[parent[0], parent[1]] = 1
    im = sk.filters.gaussian(im, sigma=5)
    parent_label = sk.measure.label(im > 0.7 * im.mean())
    # Change the id of the parent so it is distinct from daughters
    parent_label[parent_label == 1] = parent_id

    # Calculate position of daughters in the first frame
    width = sk.measure.regionprops(parent_label)[0].axis_minor_length
    shift = int(width / 3)
    d1 = [parent[0] - shift, parent[1]]
    d2 = [parent[0] + shift, parent[1]]

    # Make first frame images
    im = np.zeros((img_size, img_size))
    im[d1[0], d1[1]] = 1
    im[d2[0], d2[1]] = 1

    im = sk.filters.gaussian(im, sigma=4)
    separate = sk.measure.label(im > 20 * im.mean())
    merged = sk.measure.label(im > 5 * im.mean())
    # Change merged id to match parent label
    merged[merged == 1] = parent_id

    # Move daughters again for the last frame
    d1 = [d1[0] - shift, d1[1]]
    d2 = [d2[0] + shift, d2[1]]

    im = np.zeros((img_size, img_size))
    im[d1[0], d1[1]] = 1
    im[d2[0], d2[1]] = 1

    im = sk.filters.gaussian(im, sigma=5)
    last = sk.measure.label(im > 0.7 * im.mean())

    early_div = np.stack([parent_label, separate, last])
    late_div = np.stack([parent_label, merged, last])

    # Generate early division graph
    Ge = nx.DiGraph()
    Ge.add_edge('10_0', '1_1')
    Ge.add_edge('10_0', '2_1')
    Ge.nodes['10_0']['division'] = True

    Ge.add_edge('1_1', '1_2')
    Ge.add_edge('2_1', '2_2')

    # Generate late division graph
    Gl = nx.DiGraph()
    Gl.add_edge('10_0', '10_1')

    Gl.add_edge('10_1', '1_2')
    Gl.add_edge('10_1', '2_2')
    Gl.nodes['10_1']['division'] = True

    return early_div, late_div, Ge, Gl
