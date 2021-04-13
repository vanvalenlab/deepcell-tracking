# Copyright 2016-2021 David Van Valen at California Institute of Technology
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

import json
import os
import re
import tarfile
import tempfile
from io import BytesIO

import cv2
import numpy as np
from skimage import transform

from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

from scipy.spatial.distance import cdist

from deepcell_toolbox.utils import resize


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


def load_trks(filename):
    """Load a trk/trks file.

    Args:
        filename (str): full path to the file including .trk/.trks.

    Returns:
        dict: A dictionary with raw, tracked, and lineage data.
    """
    with tarfile.open(filename, 'r') as trks:

        # numpy can't read these from disk...
        array_file = BytesIO()
        array_file.write(trks.extractfile('raw.npy').read())
        array_file.seek(0)
        raw = np.load(array_file)
        array_file.close()

        array_file = BytesIO()
        array_file.write(trks.extractfile('tracked.npy').read())
        array_file.seek(0)
        tracked = np.load(array_file)
        array_file.close()

        # trks.extractfile opens a file in bytes mode, json can't use bytes.
        _, file_extension = os.path.splitext(filename)

        if file_extension == '.trks':
            trk_data = trks.getmember('lineages.json')
            lineages = json.loads(trks.extractfile(trk_data).read().decode())
            # JSON only allows strings as keys, so convert them back to ints
            for i, tracks in enumerate(lineages):
                lineages[i] = {int(k): v for k, v in tracks.items()}

        elif file_extension == '.trk':
            trk_data = trks.getmember('lineage.json')
            lineage = json.loads(trks.extractfile(trk_data).read().decode())
            # JSON only allows strings as keys, so convert them back to ints
            lineages = []
            lineages.append({int(k): v for k, v in lineage.items()})

    return {'lineages': lineages, 'X': raw, 'y': tracked}


def trk_folder_to_trks(dirname, trks_filename):
    """Compiles a directory of trk files into one trks_file.

    Args:
        dirname (str): full path to the directory containing multiple trk files.
        trks_filename (str): desired filename (the name should end in .trks).
    """
    lineages = []
    raw = []
    tracked = []

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    file_list = os.listdir(dirname)
    file_list_sorted = sorted(file_list, key=alphanum_key)

    for filename in file_list_sorted:
        trk = load_trks(os.path.join(dirname, filename))
        lineages.append(trk['lineages'][0])  # this is loading a single track
        raw.append(trk['X'])
        tracked.append(trk['y'])

    file_path = os.path.join(os.path.dirname(dirname), trks_filename)

    save_trks(file_path, lineages, raw, tracked)


def save_trks(filename, lineages, raw, tracked):
    """Saves raw, tracked, and lineage data into one trks_file.

    Args:
        filename (str): full path to the final trk files.
        lineages (dict): a list of dictionaries saved as a json.
        raw (np.array): raw images data.
        tracked (np.array): annotated image data.

    Raises:
        ValueError: filename does not end in ".trks".
    """
    if not str(filename).lower().endswith('.trks'):
        raise ValueError('filename must end with `.trks`. Found %s' % filename)

    with tarfile.open(filename, 'w') as trks:
        with tempfile.NamedTemporaryFile('w', delete=False) as lineages_file:
            json.dump(lineages, lineages_file, indent=4)
            lineages_file.flush()
            lineages_file.close()
            trks.add(lineages_file.name, 'lineages.json')
            os.remove(lineages_file.name)

        with tempfile.NamedTemporaryFile(delete=False) as raw_file:
            np.save(raw_file, raw)
            raw_file.flush()
            raw_file.close()
            trks.add(raw_file.name, 'raw.npy')
            os.remove(raw_file.name)

        with tempfile.NamedTemporaryFile(delete=False) as tracked_file:
            np.save(tracked_file, tracked)
            tracked_file.flush()
            tracked_file.close()
            trks.add(tracked_file.name, 'tracked.npy')
            os.remove(tracked_file.name)


def trks_stats(filename):
    """For a given trks_file, find the Number of cell tracks,
       the Number of frames per track, and the Number of divisions.

    Args:
        filename (str): full path to a trks file.

    Raises:
        ValueError: filename is not a .trk or .trks file.
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in {'.trks', '.trk'}:
        raise ValueError('`trks_stats` expects a .trk or .trks but found a ' +
                         str(ext))

    training_data = load_trks(filename)
    X = training_data['X']
    y = training_data['y']
    daughters = [{cell: fields['daughters']
                  for cell, fields in tracks.items()}
                 for tracks in training_data['lineages']]

    print('Dataset Statistics: ')
    print('Image data shape: ', X.shape)
    print('Number of lineages (should equal batch size): ',
          len(training_data['lineages']))

    # Calculate cell density
    frame_area = X.shape[2] * X.shape[3]

    avg_cells_in_frame = []
    for batch in range(y.shape[0]):
        num_cells_in_frame = []
        for frame in y[batch]:
            cells_in_frame = len(np.unique(frame)) - 1  # unique returns 0 (BKGD)
            num_cells_in_frame.append(cells_in_frame)
        avg_cells_in_frame.append(np.average(num_cells_in_frame))
    avg_cells_per_sq_pixel = np.average(avg_cells_in_frame) / frame_area

    # Calculate division information
    total_tracks = 0
    total_divisions = 0
    avg_frame_counts_in_batches = []
    for batch, daughter_batch in enumerate(daughters):
        num_tracks_in_batch = len(daughter_batch)
        num_div_in_batch = len([c for c in daughter_batch if daughter_batch[c]])
        total_tracks = total_tracks + num_tracks_in_batch
        total_divisions = total_divisions + num_div_in_batch
        frame_counts = []
        for cell_id in daughter_batch.keys():
            frame_count = 0
            for frame in y[batch]:
                cells_in_frame = np.unique(frame)
                if cell_id in cells_in_frame:
                    frame_count += 1
            frame_counts.append(frame_count)
        avg_frame_counts_in_batches.append(np.average(frame_counts))
    avg_num_frames_per_track = np.average(avg_frame_counts_in_batches)

    print('Total number of unique tracks (cells)      - ', total_tracks)
    print('Total number of divisions                  - ', total_divisions)
    print('Average cell density (cells/100 sq pixels) - ', avg_cells_per_sq_pixel * 100)
    print('Average number of frames per track         - ', int(avg_num_frames_per_track))


def get_max_cells(y):
    """Helper function for finding the maximum number of cells in a frame of a movie, across
    all frames of the movie. Can be used for batches/tracks interchangeably with frames/cells.

    Args:
        y (np.array): Annotated image data

    Returns:
        max_cells (int): The maximum number of cells in any frame
    """
    max_cells = 0
    for frame in range(y.shape[0]):
        cells = np.unique(y[frame])
        n_cells = cells[cells != 0].shape[0]
        if n_cells > max_cells:
            max_cells = n_cells
    return max_cells


def normalize_adj_matrix(adj, epsilon=1e-5):
    # Normalize the adjacency matrix

    normed_adj = np.zeros(adj.shape, dtype='float32')
    for t in range(adj.shape[-1]):
        adj_frame = adj[..., t]
        # setup degree matrix
        degree_matrix = np.zeros(adj_frame.shape, dtype=np.float32)
        # determine whether multiple batches being normalized
        if len(adj.shape) == 4:
            # adj is (batch, node, node, time)
            degrees = np.sum(adj_frame, axis=1)
            for batch, degree in enumerate(degrees):
                degree = (degree + epsilon) ** -0.5
                degree_matrix[batch] = np.diagflat(degree)

        elif len(adj.shape) == 3:
            # adj is (cells, cells, time)
            norm_adj = np.matmul(degree_matrix, adj_frame)
            norm_adj = np.matmul(norm_adj, degree_matrix)
            normed_adj[..., t] = norm_adj

        else:
            raise ValueError('Only 3 & 4 dim adjacency matrices are supported')

    return normed_adj


# TODO: This class contains code that could be reused for tracking.py
#       The only difference is usually doing things by batch vs frame
class Track(object):
    def __init__(self,
                 path,
                 appearance_dim=32,
                 distance_threshold=64):

        training_data = load_trks(path)
        self.X = training_data['X'].astype(np.float32)
        self.y = training_data['y'].astype(np.int)
        self.lineages = training_data['lineages']
        self.appearance_dim = appearance_dim
        self.distance_threshold = distance_threshold

        # Correct lineages
        self._correct_lineages()

        # Remove bad batches
        self._check_lineages()

        # Create feature dictionaries
        features_dict = self._get_features()
        self.appearances = features_dict['appearances']
        self.morphologies = features_dict['morphologies']
        self.centroids = features_dict['centroids']
        self.adj_matrix = features_dict['adj_matrix']
        self.norm_adj_matrix = features_dict['norm_adj_matrix']
        self.temporal_adj_matrix = features_dict['temporal_adj_matrix']
        self.mask = features_dict['mask']
        self.track_length = features_dict['track_length']

    def _correct_lineages(self):
        n_batches = self.y.shape[0]

        # Ensure sequential labels
        new_lineages = {}
        for batch in range(n_batches):
            y_batch = self.y[batch]
            y_relabel, fw, inv = relabel_sequential(y_batch)

            new_lineages[batch] = {}

            cell_ids = np.unique(y_batch)
            cell_ids = cell_ids[cell_ids != 0]
            for cell_id in cell_ids:
                new_lineages[batch][fw[cell_id]] = {}

                # Fix label
                new_lineages[batch][fw[cell_id]]['label'] = fw[cell_id]

                # Fix parent
                parent = self.lineages[batch][cell_id]['parent']
                if parent is not None:
                    new_lineages[batch][fw[cell_id]]['parent'] = fw[parent]
                else:
                    new_lineages[batch][fw[cell_id]]['parent'] = None

                # Fix daughters
                daughters = self.lineages[batch][cell_id]['daughters']
                if len(daughters) > 0:
                    new_lineages[batch][fw[cell_id]]['daughters'] = [fw[d] for d in daughters]
                else:
                    new_lineages[batch][fw[cell_id]]['daughters'] = []

                # Fix frames
                y_true = np.sum(y_batch == cell_id, axis=(1, 2))
                y_index = np.where(y_true > 0)[0]
                new_lineages[batch][fw[cell_id]]['frames'] = list(y_index)

            self.y[batch] = y_relabel
        self.lineages = new_lineages

    def _check_lineages(self):
        # Make sure that mother cells leave the fov
        # and daughter cells enter

        bad_batch = []

        n_batches = self.y.shape[0]

        for batch in range(n_batches):
            for cell_id in self.lineages[batch].keys():

                # Get parent frames
                parent_frames = self.lineages[batch][cell_id]['frames']

                # Get daughter frames
                daughters = self.lineages[batch][cell_id]['daughters']

                if len(daughters) == 0:
                    daughter_frames = None
                else:
                    daughter_frames = [self.lineages[batch][daughter]['frames']
                                       for daughter in daughters]
                    # Check that daughter's start frame is one larger than parent end frame
                    parent_end = parent_frames[-1]
                    daughters_start = [d[0] for d in daughter_frames]

                    for ds in daughters_start:
                        if ds - parent_end != 1:
                            bad_batch.append(batch)

        bad_batch = set(bad_batch)
        bad_batch = list(bad_batch)

        print(bad_batch)

        new_X = []
        new_y = []
        new_lineages = []
        for batch in range(self.X.shape[0]):
            if batch not in bad_batch:
                new_X.append(self.X[batch])
                new_y.append(self.y[batch])
                new_lineages.append(self.lineages[batch])

        new_X = np.stack(new_X, axis=0)
        new_y = np.stack(new_y, axis=0)

        self.X = new_X
        self.y = new_y
        self.lineages = new_lineages

    # TODO: Can this function be adapted to use _create_features from tracking.py?
    def _get_features(self):
        # Extract the relevant features from the label movie
        # Appearance, morphologies, centroids, and adjacency matrices

        max_tracks = get_max_cells(self.y)
        n_batches = self.X.shape[0]
        n_frames = self.X.shape[1]
        n_channels = self.X.shape[-1]

        appearances = np.zeros((n_batches,
                                max_tracks,
                                n_frames,
                                self.appearance_dim,
                                self.appearance_dim,
                                n_channels), dtype=np.float32)

        morphologies = np.zeros((n_batches,
                                 max_tracks,
                                 n_frames,
                                 3), dtype=np.float32)

        centroids = np.zeros((n_batches,
                              max_tracks,
                              n_frames,
                              2), dtype=np.float32)

        adj_matrix = np.zeros((n_batches,
                               max_tracks,
                               max_tracks,
                               n_frames), dtype=np.float32)

        temporal_adj_matrix = np.zeros((n_batches,
                                        max_tracks,
                                        max_tracks,
                                        n_frames - 1,
                                        3), dtype=np.float32)

        mask = np.zeros((n_batches,
                         max_tracks,
                         n_frames), dtype=np.float32)

        track_length = np.zeros((n_batches,
                                 max_tracks,
                                 2), dtype=np.int32)

        for batch in range(n_batches):
            for frame in range(n_frames):
                y = self.y[batch, frame, ..., 0]
                props = regionprops(y)
                for prop in props:
                    track_id = prop.label - 1

                    # Get centroid
                    centroids[batch, track_id, frame] = np.array(prop.centroid)

                    # Get morphology
                    morphologies[batch, track_id, frame] = np.array([prop.area,
                                                                     prop.perimeter,
                                                                     prop.eccentricity])

                    # Get appearance
                    minr, minc, maxr, maxc = prop.bbox
                    appearance = np.copy(self.X[batch, frame, minr:maxr, minc:maxc, :])
                    resize_shape = (self.appearance_dim, self.appearance_dim)
                    appearances[batch, track_id, frame] = resize(appearance, resize_shape)

                    # Get mask
                    mask[batch, track_id, frame] = 1

                # Get adjacency matrix
                cent = centroids[batch, :, frame, :]
                distance = cdist(cent, cent, metric='euclidean') < self.distance_threshold
                adj_matrix[batch, :, :, frame] = distance.astype(np.float32)

            # Get track length
            labels = np.unique(self.y[batch, ..., 0])
            labels = labels[labels != 0]
            for label in labels:
                track_id = label - 1
                start_frame = self.lineages[batch][label]['frames'][0]
                end_frame = self.lineages[batch][label]['frames'][-1]

                track_length[batch, track_id, 0] = start_frame
                track_length[batch, track_id, 1] = end_frame

            # Get temporal adjacency matrix
            for label in labels:
                track_id = label - 1

                # Assign same
                frames = self.lineages[batch][label]['frames']
                frames_0 = frames[0:-1]
                frames_1 = frames[1:]
                for frame_0, frame_1 in zip(frames_0, frames_1):
                    if frame_1 - frame_0 == 1:
                        temporal_adj_matrix[batch, track_id, track_id, frame_0, 0] = 1

                # Assign daughter
                daughters = self.lineages[batch][label]['daughters']
                last_frame = frames[-1]

                # WARNING: This wont work if there's a time gap between mother
                # cell disappearing and daughter cells appearing
                if len(daughters) > 0:
                    for daughter in daughters:
                        daughter_id = daughter - 1
                        temporal_adj_matrix[batch, track_id, daughter_id, last_frame, 2] = 1
                        # temporal_adj_matrix[batch, daughter_id, track_id, last_frame, 2] = 1

            # Assign different
            temporal_adj_matrix[batch, ..., 1] = (1 -
                                                  temporal_adj_matrix[batch, ..., 0] -
                                                  temporal_adj_matrix[batch, ..., 2])

            # Identify padding
            track_ids = [label - 1 for label in labels]
            for i in range(temporal_adj_matrix.shape[1]):
                if i not in track_ids:
                    temporal_adj_matrix[batch, i, ...] = -1
                    temporal_adj_matrix[batch, :, i, ...] = -1

        # Normalize adj matrix
        norm_adj_matrix = normalize_adj_matrix(adj_matrix)

        feature_dict = {}
        feature_dict['adj_matrix'] = adj_matrix
        feature_dict['norm_adj_matrix'] = norm_adj_matrix
        feature_dict['appearances'] = appearances
        feature_dict['morphologies'] = morphologies
        feature_dict['centroids'] = centroids
        feature_dict['temporal_adj_matrix'] = temporal_adj_matrix
        feature_dict['mask'] = mask
        feature_dict['track_length'] = track_length

        return feature_dict
