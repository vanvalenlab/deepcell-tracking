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
"""A cell tracking class capable of extending labels across sequential frames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import logging
import os
import pathlib
import tarfile
import tempfile
import timeit

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

import networkx as nx
import pandas as pd

from deepcell_tracking.utils import clean_up_annotations
from deepcell_tracking.utils import get_max_cells
from deepcell_tracking.utils import normalize_adj_matrix
from deepcell_tracking.utils import get_image_features
from deepcell_tracking.trk_io import save_trk


class CellTracker(object):  # pylint: disable=useless-object-inheritance
    """Solves the linear assingment problem to build a cell lineage graph.

    Args:
        movie (np.array): raw time series movie of cells.
        annotation (np.array): the labeled cell movie.
        model (keras.Model): tracking model to determine if two cells are the
            same, different, or parent/daughter.
        features (list): list of strings for the features to use.
        crop_dim (int): crop size for the appearance feature.
        death (float): paramter used to fill the death matrix in the LAP,
            (top right of the cost matrix).
        birth (float): paramter used to fill the birth matrix in the LAP,
            (bottom left of the cost matrix).
        division (float): probability threshold for assigning daughter cells.
        distance_threshold (int): maximum distance to compare cells with the model.
        track_length (int): the track length used for the model.
        neighborhood_scale_size (int): neighborhood feature size to pass to the
            model.
        neighborhood_true_size (int): original size of the neighborhood feature
            which will be scaled down to neighborhood_scale_size.
        dtype (str): data type for features, can be 'float32', 'float16', etc.
        data_format (str): determines the order of the channel axis,
            one of 'channels_first' and 'channels_last'.
    """

    def __init__(self,
                 movie,
                 annotation,
                 tracking_model,
                 neighborhood_encoder=None,
                 distance_threshold=64,
                 appearance_dim=32,
                 death=0.99,
                 birth=0.99,
                 division=0.9,
                 track_length=5,
                 embedding_axis=0,
                 dtype='float32',
                 data_format='channels_last'):

        if not len(movie.shape) == 4 or not len(annotation.shape) == 4:
            raise ValueError('Input data and labels must be rank 4 (frames, x, y, channels). '
                             'Got rank {} (X) and rank {} (y).'.format(
                                 len(movie.shape), len(annotation.shape)))

        if not movie.shape[:-1] == annotation.shape[:-1]:
            raise ValueError('Input data and labels should have the same shape'
                             ' except for the channel dimension.  Got {} and '
                             '{}'.format(movie.shape, annotation.shape))

        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('The `data_format` argument must be one of '
                             '"channels_first", "channels_last". Received: ' +
                             str(data_format))

        self.X = copy.copy(movie)
        self.y = copy.copy(annotation)
        self.tracks = {}

        self.neighborhood_encoder = neighborhood_encoder
        self.tracking_model = tracking_model
        self.distance_threshold = distance_threshold
        self.appearance_dim = appearance_dim
        self.death = death
        self.birth = birth
        self.division = division
        self.dtype = dtype
        self.track_length = track_length
        self.embedding_axis = embedding_axis

        self.a_matrix = []
        self.c_matrix = []
        self.assignments = []

        self.data_format = data_format
        self.channel_axis = 0 if data_format == 'channels_first' else -1
        self.time_axis = 1 if data_format == 'channels_first' else 0
        self.logger = logging.getLogger(str(self.__class__.__name__))

        # Clean up annotations
        self.y = clean_up_annotations(self.y, data_format=self.data_format)

        # Accounting for 0 (background) label with 0-indexing for tracks
        self.id_to_idx = {}  # int: int mapping
        self.idx_to_id = {}  # (frame, cell_idx): cell_id mapping

        # Establish features for every instance of every cell in the movie
        adj_matrices, appearances, morphologies, centroids = self._est_feats()

        # Compute embeddings for every instance of every cell in the movie
        embeddings = self._get_neighborhood_embeddings(
            appearances=appearances,
            morphologies=morphologies,
            centroids=centroids,
            adj_matrices=adj_matrices)

        # TODO: immutable dict for safety? these values should never change.
        self.features = {
            'embedding': embeddings,
            'centroid': centroids,
        }

    def _get_frame(self, tensor, frame):
        """Helper function for fetching a frame of a tensor.

        Useful for avoiding duplication of the data_format conditional.

        Args:
            tensor (np.array): The 3D tensor to slice.
            frame (int): The frame to slice out of the tensor.

        Returns:
            np.array: the 2D slice of the 3D tensor.
        """
        if self.data_format == 'channels_first':
            return tensor[:, frame]
        return tensor[frame]

    def _get_cells_in_frame(self, frame):
        """Find the labels of cells in the given frame.

        Args:
            frame (int): Frame of interest.

        Returns:
            list: All cell labels in the frame.
        """
        cells = np.unique(self._get_frame(self.y, frame))
        cells = np.delete(cells, np.where(cells == 0))  # remove the background
        return list(cells)

    def _est_feats(self):
        """
        Extract the relevant features from the label movie
        Appearance, morphologies, centroids, and adjacency matrices
        """
        max_cells = get_max_cells(self.y)
        n_frames = self.X.shape[0]
        n_channels = self.X.shape[-1]

        appearances = np.zeros((n_frames,
                                max_cells,
                                self.appearance_dim,
                                self.appearance_dim,
                                n_channels), dtype=np.float32)

        morphologies = np.zeros((n_frames, max_cells, 3), dtype=np.float32)

        centroids = np.zeros((n_frames, max_cells, 2), dtype=np.float32)

        adj_matrix = np.zeros((n_frames, max_cells, max_cells),
                              dtype=np.float32)

        for frame in range(n_frames):

            frame_features = get_image_features(
                self.X[frame], self.y[frame],
                appearance_dim=self.appearance_dim)

            for cell_idx, cell_id in enumerate(frame_features['labels']):
                self.id_to_idx[cell_id] = cell_idx
                self.idx_to_id[(frame, cell_idx)] = cell_id

            num_tracks = len(frame_features['labels'])
            centroids[frame, :num_tracks] = frame_features['centroids']
            morphologies[frame, :num_tracks] = frame_features['morphologies']
            appearances[frame, :num_tracks] = frame_features['appearances']

            cent = centroids[frame]
            distance = cdist(cent, cent, metric='euclidean') < self.distance_threshold
            adj_matrix[frame] = distance.astype('float32')

        # Normalize adj matrix
        norm_adj_matrices = normalize_adj_matrix(adj_matrix)

        return norm_adj_matrices, appearances, morphologies, centroids

    def _get_neighborhood_embeddings(self, appearances, morphologies,
                                     centroids, adj_matrices):
        """Compute the embeddings using the neighborhood encoder"""
        # Build input dictionary for neighborhood encoder model
        inputs = {
            'encoder_app_input': appearances,
            'encoder_morph_input': morphologies,
            'encoder_centroid_input': centroids,
            'encoder_adj_input': adj_matrices,
        }

        # TODO: current model doesnt organize outputs according to ordered list
        #       patching with embedding_axis
        embeddings = self.neighborhood_encoder.predict(inputs)[self.embedding_axis]
        embeddings = np.array(embeddings)
        return embeddings

    def _validate_feature_name(self, feature_name):
        if feature_name not in self.features:
            raise ValueError('{} is an invalid feature name. '
                             'Use one of embedding or centroid'.format(
                                 feature_name))

    def _get_feature(self, frame, cell_id, feature_name='embedding'):
        """Get the feature for a cell in the frame"""
        self._validate_feature_name(feature_name)
        cell_idx = self.id_to_idx[cell_id]
        return self.features[feature_name][frame, cell_idx, :]

    def _get_frame_features(self, frame, feature_name='embedding'):
        """Get the feature for the specified cells in a frame"""
        self._validate_feature_name(feature_name)

        cells_in_frame = self._get_cells_in_frame(frame)

        frame_features = {}
        for cell_id in cells_in_frame:
            f = self._get_feature(frame, cell_id, feature_name=feature_name)
            frame_features[cell_id] = f
        return frame_features

    def _create_new_track(self, frame, old_label):
        """
        This function creates new tracks
        """
        track_id = len(self.tracks)
        new_label = track_id + 1
        embedding = self._get_feature(frame, old_label, feature_name='embedding')
        centroid = self._get_feature(frame, old_label, feature_name='centroid')

        embedding = np.expand_dims(embedding, axis=0)
        centroid = np.expand_dims(centroid, axis=0)

        self.tracks[track_id] = {
            'label': new_label,
            'frames': [frame],
            'frame_labels': [old_label],
            'daughters': [],
            'capped': False,
            'frame_div': None,
            'parent': None,
            'embedding': embedding,
            'centroid': centroid
        }

        if frame > 0 and np.any(self._get_frame(self.y, frame) == new_label):
            raise Exception('new_label already in annotated frame and frame > 0')

        if self.data_format == 'channels_first':
            self.y[:, frame][self.y[:, frame] == old_label] = new_label
        else:
            self.y[frame][self.y[frame] == old_label] = new_label

    def _initialize_tracks(self):
        """Intialize the tracks. Tracks are stored in a dictionary.
        """
        frame = 0  # initial frame
        cell_ids = self._get_cells_in_frame(frame)

        for cell_id in cell_ids:
            self._create_new_track(frame, cell_id)

        # Start a tracked label array
        # TODO: This could be a source of error!
        # we are assigning a pointer not instantiating a new object
        self.y_tracked = self.y[[frame]].astype('int32')

    def _fetch_tracked_features(self, before_frame=None, feature_name='embedding'):
        """Get feature data from each tracked frame less than before_frame.

        Args:
            before_frame (int, optional): The maximum frame to from which to
                fetch feature data.
            feature_name (str): Name of feature to fetch from tracked data.

        Returns:
            dict: dictionary of feature name to np.array of feature data.
        """
        self._validate_feature_name(feature_name)

        if before_frame is None:
            before_frame = self.X.shape[0] + 1  # all frames

        track_valid_frames = ((n, [f for f in d['frames'] if f < before_frame])
                              for n, d in self.tracks.items())
        tracks_with_frames = [(n, f) for n, f in track_valid_frames if len(f) > 0]

        tracked_features = {}
        for i, (n, valid_frames) in enumerate(tracks_with_frames):
            frame_dict = {frame: j for j, frame in enumerate(valid_frames)}
            frames = valid_frames[-self.track_length:]

            if len(frames) != self.track_length:
                # Pad the the frames with the last frame if not enough
                num_missing = self.track_length - len(frames)
                frames = frames + [frames[-1]] * num_missing

            # Get the feature data from the identified frames
            fetched = self.tracks[n][feature_name][[frame_dict[f] for f in frames]]
            tracked_features[i] = fetched

        return tracked_features

    def _build_cost_matrix(self, assignment_matrix):
        """Build the full cost matrix based on the assignment_matrix.

        Args:
            assignment_matrix (np.array): assignment_matrix.

        Returns:
            numpy.array: cost matrix.
        """
        # Initialize cost matrix
        num_tracks, num_cells = assignment_matrix.shape
        cost_matrix = np.zeros((num_tracks + num_cells,) * 2, dtype=self.dtype)

        # assignment matrix - top left
        cost_matrix[0:num_tracks, 0:num_cells] = assignment_matrix

        # birth matrix - bottom left
        birth_diagonal = np.array([self.birth] * num_cells)
        birth_matrix = np.zeros((num_cells, num_cells), dtype=self.dtype)
        birth_matrix = np.diag(birth_diagonal) + np.ones(birth_matrix.shape)
        birth_matrix = birth_matrix - np.eye(num_cells)
        cost_matrix[num_tracks:, 0:num_cells] = birth_matrix

        # death matrix - top right
        death_matrix = np.ones((num_tracks, num_tracks), dtype=self.dtype)
        death_matrix = self.death * np.eye(num_tracks) + death_matrix
        death_matrix = death_matrix - np.eye(num_tracks)
        cost_matrix[0:num_tracks, num_cells:] = death_matrix

        # mordor matrix - bottom right
        cost_matrix[num_tracks:, num_cells:] = assignment_matrix.T
        return cost_matrix

    def _get_cost_matrix(self, frame):
        """Use the model predictions to build an assignment matrix to be solved.

        Args:
            frame (int): The frame with cells to assign.

        Returns:
            tuple: the assignment matrix and the predictions used to build it.
        """
        inputs = {}
        relevant_tracks = []
        for feature_name in self.features:
            # Get the embeddings for previously tracked cells
            current_feature = self._fetch_tracked_features(
                before_frame=frame, feature_name=feature_name)

            # Get the embeddings for the current frame
            future_feature = self._get_frame_features(
                frame=frame, feature_name=feature_name)

            if not relevant_tracks:
                for track_id in current_feature:
                    relevant_tracks.append(track_id)

            # Convert from dict to arrays
            current_feature_arr = np.stack([
                current_feature[k] for k in current_feature
            ], axis=1)  # time axis already included
            future_feature_arr = np.stack([
                future_feature[k] for k in future_feature
            ], axis=0)

            # Add time dimension
            future_feature_arr = np.expand_dims(future_feature_arr, axis=0)

            # Add batch dimension
            current_feature_arr = np.expand_dims(current_feature_arr, axis=0)
            future_feature_arr = np.expand_dims(future_feature_arr, axis=0)

            # Add feature to inputs
            # model expects "current_embeddings" but feature_name is "embedding"
            # TODO: this name is hardcoded based on a model from deepcell-tf
            inputs['current_{}s'.format(feature_name)] = current_feature_arr
            inputs['future_{}s'.format(feature_name)] = future_feature_arr

        t = timeit.default_timer()

        # Perform inference
        predictions = self.tracking_model.predict(inputs)
        predictions = predictions[0, 0, ...]  # Remove the batch and time dimension
        assignment_matrix = 1 - predictions[..., 0]

        for track_id in relevant_tracks:
            if self.tracks[track_id]['capped']:
                assignment_matrix[track_id, :] = 1

        self.a_matrix.append(predictions)

        # Assemble full cost matrix
        cost_matrix = self._build_cost_matrix(assignment_matrix)
        self.logger.debug('Built cost matrix for frame %s in %s s.',
                          frame, timeit.default_timer() - t)
        self.c_matrix.append(cost_matrix)

        predictions_dict = {}
        predictions_dict['predictions'] = predictions
        predictions_dict['track_ids'] = relevant_tracks

        return cost_matrix, predictions_dict

    def _update_tracks(self, assignments, frame, predictions):
        """Update the graph based on the assignment matrix for the frame.

        Use the assignment matrix to determine if each cell in the frame.
        belongs to a track or is a daughter of a track, and updates the graph
        accordingly.

        Args:
            assignments (np.array): completed assignment matrix used to assign
                cells to existing tracks.
            frame (int): the frame of cells to assign.
            predictions (dict): dictionary of trackID-cellID combination,
                and the probability they are the same cell.
        """
        t = timeit.default_timer()
        cells_in_frame = self._get_cells_in_frame(frame)

        # Number of lables present in the current frame (needed to build cost matrix)
        y_tracked_update = np.zeros((1, self.y.shape[1], self.y.shape[2], 1), dtype='int32')

        self.assignments.append(assignments)

        for a in range(assignments.shape[0]):
            track, cell = assignments[a]

            # map the index from the LAP assignment to the cell label
            try:
                cell_id = cells_in_frame[cell]
            except IndexError:
                # cell has "died" or is a shadow assignment
                # no assignment should be made
                continue

            cell_embedding = self._get_feature(frame, cell_id, feature_name='embedding')
            cell_centroid = self._get_feature(frame, cell_id, feature_name='centroid')

            cell_embedding = np.expand_dims(cell_embedding, axis=0)
            cell_centroid = np.expand_dims(cell_centroid, axis=0)

            if track in self.tracks:  # Add cell and frame to track
                self.tracks[track]['frames'].append(frame)
                self.tracks[track]['frame_labels'].append(cell_id)
                self.tracks[track]['embedding'] = np.concatenate([
                    self.tracks[track]['embedding'],
                    cell_embedding
                ], axis=0)
                self.tracks[track]['centroid'] = np.concatenate([
                    self.tracks[track]['centroid'],
                    cell_centroid
                ], axis=0)

                # Labels and indices differ by 1
                y_tracked_update[self.y[[frame]] == cell_id] = track + 1
                self.y[frame][self.y[frame] == cell_id] = track + 1

            else:  # Create a new track if there was a birth
                self._create_new_track(frame, cell_id)
                new_track_id = max(self.tracks)
                new_label = new_track_id + 1
                self.logger.info('Created new track for cell %s.', new_label)

                # See if the new track has a parent
                parent = self._get_parent(frame, cell_id, predictions)
                if parent is not None:
                    self.logger.info('Detected division! Cell %s is daughter '
                                     'of cell %s.', new_label, parent + 1)
                    self.tracks[new_track_id]['parent'] = parent
                    self.tracks[parent]['daughters'].append(new_track_id)
                else:
                    self.tracks[new_track_id]['parent'] = None

                y_tracked_update[self.y[[frame]] == new_label] = new_track_id + 1
                self.y[frame][self.y[frame] == new_label] = new_track_id + 1

        # Check and make sure cells that divided did not get assigned to the same cell
        for track in range(len(self.tracks)):
            if not self.tracks[track]['daughters']:
                continue  # Filter out cells that have not divided

            # Cap tracks for any divided cells
            if not self.tracks[track]['capped']:
                self.tracks[track]['frame_div'] = int(frame)
                self.tracks[track]['capped'] = True

            try:
                frame_idx = self.tracks[track]['frames'].index(frame)
            except ValueError:
                continue  # Filter out tracks that are not in the frame

            # Create new track
            new_track_id = len(self.tracks)
            new_label = new_track_id + 1
            self._create_new_track(frame, self.tracks[track]['frame_labels'][-1])
            self.tracks[new_track_id]['parent'] = track

            # Remove features and frame from old track
            del self.tracks[track]['frames'][frame_idx]
            del self.tracks[track]['frame_labels'][frame_idx]
            self.tracks[track]['embedding'] = np.delete(
                self.tracks[track]['embedding'],
                frame_idx, axis=0
            )
            self.tracks[track]['centroid'] = np.delete(
                self.tracks[track]['centroid'],
                frame_idx, axis=0
            )
            self.tracks[track]['daughters'].append(new_track_id)

            # Change y_tracked_update
            old_label = self.tracks[track]['label']
            y_tracked_update[self.y[[frame]] == old_label] = new_label
            # TODO: Having y and y_tracked is redundant. only one should be changed
            self.y[frame][self.y[frame] == old_label] = new_label

        # Update the tracked label array
        self.y_tracked = np.concatenate([self.y_tracked, y_tracked_update], axis=0)
        self.logger.debug('Updated tracks for frame %s in %s s.',
                          frame, timeit.default_timer() - t)

    def _get_parent(self, frame, cell, predictions):
        """Searches the tracks for the parent of a given cell.

        Args:
            frame (int): the frame the cell appears in.
            cell (int): the label of the cell in the frame.
            predictions (dict): dictionary of trackID-cellID combination,
                and the probability they are the same cell.

        Returns:
            int: The parent cell's id or None if no parent exists.
        """
        parent_id = None
        max_prob = self.division

        track_ids = predictions['track_ids']
        predictions = predictions['predictions']

        for track_idx, track_id in enumerate(track_ids):
            # Make sure capped tracks can't be assigned parents
            if self.tracks[track_id]['capped']:
                continue

            for cell_idx in range(predictions.shape[1]):
                cell_id = self.idx_to_id[(frame, cell_idx)]

                if cell_id == cell:
                    # Do not call a newly-appeared sibling of "cell" a parent
                    if self.tracks[track_id]['frames'] == [frame]:
                        continue

                    # probability cell is part of the track
                    prob = predictions[track_idx, cell_idx, 2]

                    if prob > max_prob:
                        parent_id, max_prob = track_id, prob

        return parent_id

    def _track_frame(self, frame):
        """Inner function for tracking each frame"""
        t = timeit.default_timer()
        self.logger.info('Tracking frame %s', frame)

        cost_matrix, predictions = self._get_cost_matrix(frame)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = np.stack([row_ind, col_ind], axis=1)

        self._update_tracks(assignments, frame, predictions)
        self.logger.info('Tracked frame %s in %s s.',
                         frame, timeit.default_timer() - t)

    def track_cells(self):
        """Tracks all of the cells in every frame.
        """
        start = timeit.default_timer()
        self._initialize_tracks()

        for frame in range(1, self.X.shape[self.time_axis]):
            self._track_frame(frame)

        self.logger.info('Tracked all %s frames in %s s.',
                         self.X.shape[self.time_axis],
                         timeit.default_timer() - start)

    def _track_review_dict(self):
        def process(key, track_item):
            if track_item is None:
                return track_item
            if key == 'daughters':
                return list(map(lambda x: x + 1, track_item))
            elif key == 'parent':
                return track_item + 1
            else:
                return track_item

        track_keys = ['label', 'frames', 'daughters', 'capped', 'frame_div', 'parent']

        return {'tracks': {track['label']: {key: process(key, track[key]) for key in track_keys}
                           for _, track in self.tracks.items()},
                'X': self.X,
                'y': self.y,
                'y_tracked': self.y_tracked}

    def dataframe(self, **kwargs):
        """Returns a dataframe of the tracked cells with lineage.
        Uses only the cell labels not the ids.
        _track_cells must be called first!
        """
        # possible kwargs are extra_columns
        extra_columns = ['cell_type', 'set', 'part', 'montage']
        track_columns = ['label', 'daughters', 'frame_div']

        incorrect_args = set(kwargs) - set(extra_columns)
        if incorrect_args:
            raise ValueError('Invalid argument {}'.format(incorrect_args.pop()))

        # filter extra_columns by the ones we passed in
        extra_columns = [c for c in extra_columns if c in kwargs]

        # extra_columns are the same for every row, cache the values
        extra_column_vals = [kwargs[c] for c in extra_columns if c in kwargs]

        # fill the dataframe
        data = []
        for cell_id, track in self.tracks.items():
            data.append(extra_column_vals + [track[c] for c in track_columns])
        dataframe = pd.DataFrame(data, columns=extra_columns + track_columns)

        # daughters contains track_id not labels
        dataframe['daughters'] = dataframe['daughters'].apply(
            lambda d: [self.tracks[x]['label'] for x in d])

        return dataframe

    def postprocess(self, filename=None, time_excl=9):
        """Use graph postprocessing to eliminate false positive division errors
        using a graph-based detection method. False positive errors are when a
        cell is noted as a daughter of itself before the actual division occurs.
        If a filename is passed, save the state of the cell tracker to a .trk
        ('track') file. time_excl is the minimum number of frames expected to
        exist between legitimate divisions
        """

        # Load data
        track_review_dict = self._track_review_dict()

        # Prep data
        tracked = track_review_dict['y_tracked'].astype('uint16')
        lineage = track_review_dict['tracks']

        # Identify false positives (FPs)
        G = self._track_to_graph(lineage)
        FPs = self._flag_false_pos(G, time_excl)
        FPs_candidates = sorted(FPs.items(), key=lambda v: int(v[0].split('_')[1]))
        FPs_sorted = self._review_candidate_nodes(FPs_candidates)

        # If FPs exist, use the results to correct
        while len(FPs_sorted) != 0:

            lineage, tracked = self._remove_false_pos(lineage, tracked, FPs_sorted[0])
            G = self._track_to_graph(lineage)
            FPs = self._flag_false_pos(G, time_excl)
            FPs_candidates = sorted(FPs.items(), key=lambda v: int(v[0].split('_')[1]))
            FPs_sorted = self._review_candidate_nodes(FPs_candidates)

        # Make sure the assignment is correct
        track_review_dict['y_tracked'] = tracked
        track_review_dict['tracks'] = lineage

        # Save information to a track file file if requested
        if filename is not None:
            self.dump(filename, track_review_dict)

        return track_review_dict

    def dump(self, filename, track_review_dict=None):
        """Writes the state of the cell tracker to a .trk ('track') file.
        Includes raw & tracked images, and a lineage.json for parent/daughter
        information.
        """
        if not track_review_dict:
            track_review_dict = self._track_review_dict()

        filename = pathlib.Path(filename)

        if filename.suffix != '.trk':
            filename = filename.with_suffix('.trk')

        filename = str(filename)

        save_trk(filename=filename,
                 lineage=track_review_dict['tracks'],
                 raw=track_review_dict['X'],
                 tracked=track_review_dict['y_tracked'])

    def _track_to_graph(self, tracks):
        """Create a graph from the lineage information"""
        Dattr = {}
        edges = pd.DataFrame()

        for L in tracks.values():
            # Calculate node ids
            cellid = ['{}_{}'.format(L['label'], f) for f in L['frames']]
            # Add edges from cell ids
            edges = edges.append(pd.DataFrame({'source': cellid[0:-1],
                                               'target': cellid[1:]}))

            # Collect any division attributes
            if L['frame_div'] is not None:
                Dattr['{}_{}'.format(L['label'], L['frame_div'] - 1)] = {'division': True}

            # Create any daughter-parent edges
            if L['parent'] is not None:
                source = '{}_{}'.format(L['parent'], min(L['frames']) - 1)
                target = '{}_{}'.format(L['label'], min(L['frames']))
                edges = edges.append(pd.DataFrame({'source': [source],
                                                   'target': [target]}))

        G = nx.from_pandas_edgelist(edges, source='source', target='target')
        nx.set_node_attributes(G, Dattr)
        return G

    def _flag_false_pos(self, G, time_excl):
        """Examine graph for false positive nodes
        """

        # TODO: Current implementation may eliminate some divisions at the edge of the frame -
        #       Further research needed

        # Identify false positive nodes
        node_fix = []
        for g in (G.subgraph(c) for c in nx.connected_components(G)):
            div_nodes = [n for n, d in g.nodes(data=True) if d.get('division')]
            if len(div_nodes) > 1:
                for nd in div_nodes:
                    if g.degree(nd) == 2:
                        # Check how close suspected FP is to other known divisions

                        keep_div = True
                        for div_nd in div_nodes:
                            if div_nd != nd:
                                time_spacing = abs(int(nd.split('_')[1]) -
                                                   int(div_nd.split('_')[1]))
                                # If division is sufficiently far away
                                # we should exclude it from FP list
                                if time_spacing > time_excl:
                                    keep_div = False

                        if keep_div is True:
                            node_fix.append(nd)

        # Add supplementary information for each false positive
        D = {}
        for node in node_fix:
            D[node] = {
                'false positive': node,
                'neighbors': list(G.neighbors(node)),
                'connected lineages': set([int(node.split('_')[0])
                                           for node in nx.node_connected_component(G, node)])
            }

        return D

    def _review_candidate_nodes(self, FPs_candidates):
        """ review candidate false positive nodes and remove any errant degree 2 nodes.
        """
        FPs_presort = {}
        # review candidate false positive nodes and remove any errant degree 2 nodes
        for candidate_node in FPs_candidates:
            node = candidate_node[0]
            node_info = candidate_node[1]

            neighbors = []  # structure will be [(neighbor1, frame), (neighbor2,frame)]
            for neighbor in node_info['neighbors']:
                neighbor_label = int(neighbor.split('_')[0])
                neighbor_frame = int(neighbor.split('_')[1])
                neighbors.append((neighbor_label, neighbor_frame))

            # if this cell only exists in one frame (and then it divides) but its 2 neighbors
            # both exist in the same frame it will be a degree 2 node but not be a false positive
            if neighbors[0][1] != neighbors[1][1]:
                FPs_presort[node] = node_info

        FPs_sorted = sorted(FPs_presort.items(), key=lambda v: int(v[0].split('_')[1]))

        return FPs_sorted

    def _remove_false_pos(self, lineage, tracked, FP_info):
        """ Remove nodes that have been identified as false positive divisions.
        """
        node = FP_info[0]
        node_info = FP_info[1]

        fp_label = int(node.split('_')[0])
        fp_frame = int(node.split('_')[1])

        neighbors = []  # structure will be [(neighbor1, frame), (neighbor2,frame)]
        for neighbor in node_info['neighbors']:
            neighbor_label = int(neighbor.split('_')[0])
            neighbor_frame = int(neighbor.split('_')[1])
            neighbors.append((neighbor_label, neighbor_frame))

        # Verify that the FP node only 2 neighbors - 1 before it and one after it
        if len(neighbors) == 2:
            # order the neighbors such that the time (frame order) is respected
            if neighbors[0][1] > neighbors[1][1]:
                temp = neighbors[0]
                neighbors[0] = neighbors[1]
                neighbors[1] = temp

            # Decide which labels to extend and which to remove

            # Neighbor_1 has same label as fp - the actual division hasnt occurred yet
            if fp_label == neighbors[0][0]:
                # The model mistakenly identified a division before the actual division occurred
                label_to_remove = neighbors[1][0]
                label_to_extend = neighbors[0][0]

                # Give all of the errant divisions information to the correct track
                lineage[label_to_extend]['frames'].extend(lineage[label_to_remove]['frames'])
                lineage[label_to_extend]['daughters'] = lineage[label_to_remove]['daughters']
                lineage[label_to_extend]['frame_div'] = lineage[label_to_remove]['frame_div']

                # Adjust the parent information for the actual daughters
                daughter_labels = lineage[label_to_remove]['daughters']
                for daughter in daughter_labels:
                    lineage[daughter]['parent'] = lineage[label_to_remove]['parent']

                # Remove the errant node from the annotated images
                channel = 0  # These images should only have one channel
                for frame in lineage[label_to_remove]['frames']:
                    label_loc = np.where(tracked[frame, :, :, channel] == label_to_remove)
                    tracked[frame, :, :, channel][label_loc] = label_to_extend

                # Remove the errant node from the lineage
                del lineage[label_to_remove]

            # Neighbor_2 has same label as fp - the actual division ocurred &
            # the model mistakenly allowed another
            # elif fp_label == neighbors[1][0]:
                # The model mistakenly identified a division after
                # the actual division occurred
                # label_to_remove = fp_label

            # Neither neighbor has same label as fp - the actual division
            # ocurred & the model mistakenly allowed another
            else:
                # The model mistakenly identified a division after the actual division occurred
                label_to_remove = fp_label
                label_to_extend = neighbors[1][0]

                # Give all of the errant divisions information to the correct track
                lineage[label_to_extend]['frames'] = \
                    lineage[fp_label]['frames'] + lineage[label_to_extend]['frames']
                lineage[label_to_extend]['parent'] = lineage[fp_label]['parent']

                # Adjust the parent information for the actual daughter
                parent_label = lineage[fp_label]['parent']
                for d_idx, daughter in enumerate(lineage[parent_label]['daughters']):
                    if daughter == fp_label:
                        lineage[parent_label]['daughters'][d_idx] = label_to_extend

                # Remove the errant node from the annotated images
                channel = 0  # These images should only have one channel
                for frame in lineage[label_to_remove]['frames']:
                    label_loc = np.where(tracked[frame, :, :, channel] == label_to_remove)
                    tracked[frame, :, :, channel][label_loc] = label_to_extend

                # Remove the errant node
                del lineage[label_to_remove]

        else:
            self.logger.error('Error: More than 2 neighbor nodes')

        return lineage, tracked
