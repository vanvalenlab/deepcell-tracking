# Copyright 2016-2022 Van Valen Lab at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Utilities for tracking cells"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import warnings

import numpy as np
import networkx as nx
import pandas as pd

from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

from deepcell_toolbox.utils import resize
from deepcell_toolbox import compute_overlap

from deepcell_tracking.trk_io import load_trks

# Imports for backwards compatibility
from deepcell_tracking.trk_io import save_trks, save_trk, save_track_data, trk_folder_to_trks


def clean_up_annotations(y, uid=None, data_format='channels_last'):
    """Relabels every frame in the label matrix.

    Args:
        y (np.array): annotations to relabel sequentially.
        uid (int, optional): starting ID to begin labeling cells.
        data_format (str): determines the order of the channel axis,
            one of 'channels_first' and 'channels_last'.

    Returns:
        np.array: Cleaned up annotations.
    """
    y = y.astype('int32')
    time_axis = 1 if data_format == 'channels_first' else 0
    num_frames = y.shape[time_axis]

    all_uniques = []
    for f in range(num_frames):
        cells = np.unique(y[:, f] if data_format == 'channels_first' else y[f])
        cells = np.delete(cells, np.where(cells == 0))
        all_uniques.append(cells)

    # The annotations need to be unique across all frames
    uid = sum(len(x) for x in all_uniques) + 1 if uid is None else uid
    for frame, unique_cells in zip(range(num_frames), all_uniques):
        y_frame = y[:, frame] if data_format == 'channels_first' else y[frame]
        y_frame_new = np.zeros(y_frame.shape)
        for cell_label in unique_cells:
            y_frame_new[y_frame == cell_label] = uid
            uid += 1
        if data_format == 'channels_first':
            y[:, frame] = y_frame_new
        else:
            y[frame] = y_frame_new
    return y


def count_pairs(y, same_probability=0.5, data_format='channels_last'):
    """Compute number of training samples needed to observe all cell pairs.

    Args:
        y (np.array): 5D tensor of cell labels.
        same_probability (float): liklihood that 2 cells are the same.
        data_format (str): determines the order of the channel axis,
            one of 'channels_first' and 'channels_last'.

    Returns:
        int: the total pairs needed to sample to see all possible pairings.
    """
    total_pairs = 0
    zaxis = 2 if data_format == 'channels_first' else 1
    for b in range(y.shape[0]):
        # count the number of cells in each image of the batch
        cells_per_image = []
        for f in range(y.shape[zaxis]):
            if data_format == 'channels_first':
                num_cells = len(np.unique(y[b, :, f, :, :]))
            else:
                num_cells = len(np.unique(y[b, f, :, :, :]))
            cells_per_image.append(num_cells)

        # Since there are many more possible non-self pairings than there
        # are self pairings, we want to estimate the number of possible
        # non-self pairings and then multiply that number by two, since the
        # odds of getting a non-self pairing are 50%, to find out how many
        # pairs we would need to sample to (statistically speaking) observe
        # all possible cell-frame pairs. We're going to assume that the
        # average cell is present in every frame. This will lead to an
        # underestimate of the number of possible non-self pairings, but it
        # is unclear how significant the underestimate is.
        average_cells_per_frame = sum(cells_per_image) // y.shape[zaxis]
        non_self_cellframes = (average_cells_per_frame - 1) * y.shape[zaxis]
        non_self_pairings = non_self_cellframes * max(cells_per_image)

        # Multiply cell pairings by 2 since the
        # odds of getting a non-self pairing are 50%
        cell_pairings = non_self_pairings // same_probability
        # Add this batch cell-pairings to the total count
        total_pairs += cell_pairings
    return total_pairs


def trks_stats(filename=None, X=None, y=None, lineages=None):
    """For a given trks_file, find the Number of cell tracks,
       the Number of frames per track, and the Number of divisions.

    Args:
        filename (str): full path to a trks file.

    Raises:
        ValueError: filename is not a .trk or .trks file.
    """
    if filename:
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in {'.trks', '.trk'}:
            raise ValueError(
                '`trks_stats` expects a .trk or .trks but found a {}'.format(ext))

        training_data = load_trks(filename)
        X = training_data['X']
        y = training_data['y']
        lineages = training_data['lineages']

    if not filename and not all([X is not None, y is not None, lineages is not None]):
        raise ValueError('Either filename or X, y, and lineages must be provided as input')

    print('Dataset Statistics: ')
    print('Image data shape: ', X.shape)
    print('Number of lineages (should equal batch size): ', len(lineages))

    total_tracks = 0
    total_divisions = 0

    # Calculate cell density
    frame_area = X.shape[2] * X.shape[3]

    avg_cells_in_frame = []
    avg_frame_counts_in_batches = []
    for batch in range(y.shape[0]):
        tracks = lineages[batch]
        total_tracks += len(tracks)
        num_frames_per_track = []

        for cell_lineage in tracks.values():
            num_frames_per_track.append(len(cell_lineage['frames']))
            if cell_lineage.get('daughters', []):
                total_divisions += 1
        avg_frame_counts_in_batches.append(np.average(num_frames_per_track))

        num_cells_in_frame = []
        for frame in range(len(y[batch])):
            y_frame = y[batch, frame]
            cells_in_frame = np.unique(y_frame)
            cells_in_frame = np.delete(cells_in_frame, 0)  # rm background
            num_cells_in_frame.append(len(cells_in_frame))
        avg_cells_in_frame.append(np.average(num_cells_in_frame))

    avg_cells_per_sq_pixel = np.average(avg_cells_in_frame) / frame_area
    avg_num_frames_per_track = np.average(avg_frame_counts_in_batches)

    print('Total number of unique tracks (cells)      - ', total_tracks)
    print('Total number of divisions                  - ', total_divisions)
    print('Average cell density (cells/100 sq pixels) - ', avg_cells_per_sq_pixel * 100)
    print('Average number of frames per track         - ', int(avg_num_frames_per_track))

    return {
        'n_lineages': len(lineages),
        'total_tracks': total_tracks,
        'num_div': total_divisions,
        'avg_cell_density': avg_cells_per_sq_pixel * 100
    }


def get_max_cells(y):
    """Helper function for finding the maximum number of cells in a frame of a movie, across
    all frames of the movie. Can be used for batches/tracks interchangeably with frames/cells.

    Args:
        y (np.array): Annotated image data

    Returns:
        int: The maximum number of cells in any frame
    """
    max_cells = 0
    for frame in range(y.shape[0]):
        cells = np.unique(y[frame])
        n_cells = cells[cells != 0].shape[0]
        if n_cells > max_cells:
            max_cells = n_cells
    return max_cells


def normalize_adj_matrix(adj, epsilon=1e-5):
    """Normalize the adjacency matrix

    Args:
        adj (np.array): Adjacency matrix
        epsilon (float): Used to create the degree matrix

    Returns:
        np.array: Normalized adjacency matrix

    Raises:
        ValueError: If ``adj`` has a rank that is not 3 or 4.
    """
    input_rank = len(adj.shape)
    if input_rank not in {3, 4}:
        raise ValueError('Only 3 & 4 dim adjacency matrices are supported')

    if input_rank == 3:
        # temporarily include a batch dimension for consistent processing
        adj = np.expand_dims(adj, axis=0)

    normalized_adj = np.zeros(adj.shape, dtype='float32')

    for t in range(adj.shape[1]):
        adj_frame = adj[:, t]
        # create degree matrix
        degrees = np.sum(adj_frame, axis=1)
        for batch, degree in enumerate(degrees):
            degree = (degree + epsilon) ** -0.5
            degree_matrix = np.diagflat(degree)

            normalized = np.matmul(degree_matrix, adj_frame[batch])
            normalized = np.matmul(normalized, degree_matrix)
            normalized_adj[batch, t] = normalized

    if input_rank == 3:
        # remove batch axis
        normalized_adj = normalized_adj[0]

    return normalized_adj


def relabel_sequential_lineage(y, lineage):
    """Ensure the lineage information is sequentially labeled.

    Args:
        y (np.array): Annotated z-stack of image labels.
        lineage (dict): Lineage data for y.

    Returns:
        tuple(np.array, dict): The relabeled array and corrected lineage.
    """
    y_relabel, fw, _ = relabel_sequential(y)

    new_lineage = {}

    cell_ids = np.unique(y)
    cell_ids = cell_ids[cell_ids != 0]
    for cell_id in cell_ids:
        new_cell_id = fw[cell_id]

        new_lineage[new_cell_id] = {}

        # Fix label
        # TODO: label == track ID?
        new_lineage[new_cell_id]['label'] = new_cell_id

        # Fix parent
        parent = lineage[cell_id]['parent']
        new_parent = fw[parent] if parent is not None else parent
        new_lineage[new_cell_id]['parent'] = new_parent

        # Fix daughters
        daughters = lineage[cell_id]['daughters']
        new_lineage[new_cell_id]['daughters'] = []
        for d in daughters:
            new_daughter = fw[d]
            if not new_daughter:  # missing labels get mapped to 0
                warnings.warn('Cell {} has daughter {} which is not found '
                              'in the label image `y`.'.format(cell_id, d))
            else:
                new_lineage[new_cell_id]['daughters'].append(new_daughter)

        # Fix frames
        y_true = np.any(y == cell_id, axis=(1, 2))
        y_index = y_true.nonzero()[0]
        new_lineage[new_cell_id]['frames'] = list(y_index)

    return y_relabel, new_lineage


def is_valid_lineage(y, lineage):
    """Check if a cell lineage of a single movie is valid.

    Daughter cells must exist in the frame after the parent's final frame.

    Args:
        y (numpy.array): The 3D label mask.
        lineage (dict): The cell lineages for a single movie.

    Returns:
        bool: Whether or not the lineage is valid.
    """
    all_cells = np.unique(y)
    all_cells = set([c for c in all_cells if c])

    is_valid = True

    # every lineage should have valid fields
    for cell_label, cell_lineage in lineage.items():
        # Get last frame of parent
        if cell_label not in all_cells:
            warnings.warn('Cell {} not found in the label image.'.format(
                cell_label))
            is_valid = False
            continue

        # any cells leftover are missing lineage
        all_cells.remove(cell_label)

        # validate `frames`
        y_true = np.any(y == cell_label, axis=(1, 2))
        y_index = y_true.nonzero()[0]
        frames = list(y_index)
        if frames != cell_lineage['frames']:
            warnings.warn('Cell {} has invalid frames'.format(cell_label))
            is_valid = False
            continue  # no need to test further

        last_parent_frame = cell_lineage['frames'][-1]

        for daughter in cell_lineage['daughters']:
            if daughter not in lineage:
                warnings.warn('Lineage {} has daughter {} not in lineage'.format(
                    cell_label, daughter))
                is_valid = False
                continue  # no need to test further

            # get first frame of daughter
            try:
                first_daughter_frame = lineage[daughter]['frames'][0]
            except IndexError:  # frames is empty?
                warnings.warn('Daughter {} has no frames'.format(daughter))
                is_valid = False
                continue  # no need to test further

            # Check that daughter's start frame is one larger than parent end frame
            if first_daughter_frame - last_parent_frame != 1:
                warnings.warn('Lineage {} has daughter {} in a '
                              'non-subsequent frame.'.format(
                                  cell_label, daughter))
                is_valid = False
                continue  # no need to test further

        # TODO: test parent in lineage
        parent = cell_lineage.get('parent')
        if parent:
            try:
                parent_lineage = lineage[parent]
            except KeyError:
                warnings.warn('Parent {} is not present in the lineage'.format(
                    cell_lineage['parent']))
                is_valid = False
                continue  # no need to test further
            try:
                last_parent_frame = parent_lineage['frames'][-1]
                first_daughter_frame = cell_lineage['frames'][0]
            except IndexError:  # frames is empty?
                warnings.warn('Cell {} has no frames'.format(parent))
                is_valid = False
                continue  # no need to test further
            # Check that daughter's start frame is one larger than parent end frame
            if first_daughter_frame - last_parent_frame != 1:
                warnings.warn(
                    'Cell {} ends in frame {} but daughter {} first '
                    'appears in frame {}.'.format(
                        parent, last_parent_frame, cell_label,
                        first_daughter_frame))
                is_valid = False
                continue  # no need to test further

    if all_cells:  # all cells with lineages should be removed
        warnings.warn('Cells missing their lineage: {}'.format(
            list(all_cells)))
        is_valid = False

    return is_valid  # if unchanged, all cell lineages are valid!


def get_image_features(X, y, appearance_dim=32, crop_mode='resize', norm=True):
    """Return features for every object in the array.

    Args:
        X (np.array): a 3D numpy array of raw data of shape (x, y, c).
        y (np.array): a 3D numpy array of integer labels of shape (x, y, 1).
        appearance_dim (int): The resized shape of the appearance feature.
        crop_mode (str): Whether to do a fixed crop or to crop and resize
            to create the appearance features
        norm (bool): Whether to remove non cell features and normalize the
            foreground pixels by zero-meaning and dividing by the standard
            deviation. Applies to fixed crop mode only.

    Returns:
        dict: A dictionary of feature names to np.arrays of shape
            (n, c) or (n, x, y, c) where n is the number of objects.
    """

    if crop_mode not in ['resize', 'fixed']:
        raise ValueError('crop_mode must be either resize or fixed')

    appearance_dim = int(appearance_dim)

    # each feature will be ordered based on the label.
    # labels are also stored and can be fetched by index.
    num_labels = len(np.unique(y)) - 1
    labels = np.zeros((num_labels,), dtype='int32')
    centroids = np.zeros((num_labels, 2), dtype='float32')
    morphologies = np.zeros((num_labels, 3), dtype='float32')
    appearances = np.zeros((num_labels, appearance_dim,
                            appearance_dim, X.shape[-1]), dtype='float32')

    if crop_mode == 'fixed':
        # Zero-pad the X array for fixed crop mode
        pad_width = ((appearance_dim, appearance_dim),
                     (appearance_dim, appearance_dim),
                     (0, 0))
        X_padded = np.pad(X, pad_width=pad_width)
        y_padded = np.pad(y, pad_width=pad_width)

        props = regionprops(y_padded[..., 0], cache=False)

    # iterate over all objects in y
    if crop_mode == 'resize':
        props = regionprops(y[..., 0], cache=False)

    for i, prop in enumerate(props):

        # Get label
        labels[i] = prop.label

        # Get centroid
        centroid = np.array(prop.centroid)
        centroids[i] = centroid

        # Get morphology
        morphology = np.array([
            prop.area,
            prop.perimeter,
            prop.eccentricity
        ])
        morphologies[i] = morphology

        if crop_mode == 'resize':
            # Get appearance
            minr, minc, maxr, maxc = prop.bbox
            appearance = np.copy(X[minr:maxr, minc:maxc, :])
            resize_shape = (appearance_dim, appearance_dim)
            appearance = resize(appearance, resize_shape)
            appearances[i] = appearance

        if crop_mode == 'fixed':
            cent = np.array(prop.centroid)
            delta = appearance_dim // 2
            minr = int(cent[0]) - delta
            maxr = int(cent[0]) + delta
            minc = int(cent[1]) - delta
            maxc = int(cent[1]) + delta

            app = np.copy(X_padded[minr:maxr, minc:maxc, :])
            label = np.copy(y_padded[minr:maxr, minc:maxc])

            if norm:
                # Use label as a mask to zero out non-label information
                app = app * (label == prop.label)
                idx = np.nonzero(app)

                # Check data and normalize
                if len(idx) > 0:
                    mean = np.mean(app[idx])
                    std = np.std(app[idx])
                    app[idx] = (app[idx] - mean) / std

            appearances[i] = app

    return {
        'appearances': appearances,
        'centroids': centroids,
        'labels': labels,
        'morphologies': morphologies,
    }


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


def match_nodes(gt, res, threshold=1):
    """Relabel predicted track to match GT track labels.

    Args:
        gt (np arr): label movie (y) from ground truth .trk file.
        res (np arr): label movie (y) from predicted results .trk file
        threshold (optional, float): threshold value for IoU to count as same cell. Default 1.
            If segmentations are identical, 1 works well.
            For imperfect segmentations try 0.6-0.8 to get better matching

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

    gtcells, rescells = np.where(np.nansum(iou, axis=0) >= threshold)

    return gtcells, rescells


def trk_to_graph(lineage, node_key=None):
    """Converts a lineage dictionary into a graph representation of the lineages

    Args:
        lineage (dict): Dictionary of lineage data
        node_key (dict): Map between gt nodes and result nodes

    Returns:
        networkx.Graph: Graph representation of the lineage data.
    """
    edges = []

    all_ids = set()
    single_nodes = set()
    attributes = {}

    for i, lin in lineage.items():
        # Update cell id if node_key is available
        if node_key and (i in node_key):
            idx = node_key[i]
        else:
            idx = i

        cellids = ['{}_{}'.format(idx, t) for t in lin['frames']]

        if len(cellids) == 1:
            single_nodes.add(cellids[0])

        all_ids.update(cellids)
        edges.append(pd.DataFrame({
            'source': cellids[0:-1],
            'target': cellids[1:]
        }))

        # Add connections to any daughters
        source = '{}_{}'.format(idx, max(lin['frames']))
        for d in lin['daughters']:
            # Update cell id if node_key is available
            if node_key and (i in node_key):
                d_idx = node_key[d]
            else:
                d_idx = d

            # Assume daughter appears in next frame
            target = '{}_{}'.format(d_idx, max(lin['frames']) + 1)
            edges.append(pd.DataFrame({
                'source': [source],
                'target': [target]
            }))

            attributes[source] = {'division': True}

    # Create graph
    edges = pd.concat(edges)
    G = nx.from_pandas_edgelist(edges, source='source', target='target', create_using=nx.DiGraph)
    nx.set_node_attributes(G, attributes)

    # Add all isolates to graph
    for cell_id in single_nodes:
        G.add_node(cell_id)

    return G
