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
"""A cell tracking class capable of extending labels across sequential frames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import logging
import pathlib
import tarfile
import tempfile
import timeit

import pandas as pd
import networkx as nx

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops

from deepcell_tracking.utils import resize
from deepcell_tracking.utils import clean_up_annotations


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
        max_distance (int): maximum distance to compare cells with the model.
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
                 model,
                 features={'appearance', 'distance', 'neighborhood', 'regionprop'},
                 crop_dim=32,
                 death=0.95,
                 birth=0.95,
                 division=0.9,
                 max_distance=50,
                 track_length=7,
                 neighborhood_scale_size=30,
                 neighborhood_true_size=100,
                 dtype='float32',
                 data_format='channels_last'):

        if not len(movie.shape) == 4 or not len(annotation.shape) == 4:
            raise ValueError('Input data and labels but be rank 4 '
                             '(frames, x, y, channels).  Got {} and {}.'.format(
                                 len(movie.shape), len(annotation.shape)))

        if not movie.shape[:-1] == annotation.shape[:-1]:
            raise ValueError('Input data and labels should have the same shape'
                             ' except for the channel dimension.  Got {} and '
                             '{}'.format(movie.shape, annotation.shape))

        if not features:
            raise ValueError('`features` is empty but should be a list with any'
                             ' or all of the following values: "appearance", '
                             '"distance", "neighborhood" or "regionprop".')

        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('The `data_format` argument must be one of '
                             '"channels_first", "channels_last". Received: ' +
                             str(data_format))

        self.x = copy.copy(movie)
        self.y = copy.copy(annotation)
        self.tracks = {}
        # TODO: Use a model that is served by tf-serving, not one on a local machine
        self.model = model
        self.crop_dim = crop_dim
        self.death = death
        self.birth = birth
        self.division = division
        self.max_distance = max_distance
        self.neighborhood_scale_size = neighborhood_scale_size
        self.neighborhood_true_size = neighborhood_true_size
        self.dtype = dtype
        self.data_format = data_format
        self.track_length = track_length
        self.channel_axis = 0 if data_format == 'channels_first' else -1
        self.time_axis = 1 if data_format == 'channels_first' else 0
        self.logger = logging.getLogger(str(self.__class__.__name__))

        self._track_cells = self.track_cells  # backwards compatibility

        self.features = sorted(features)

        # Clean up annotations
        self.y = clean_up_annotations(self.y, data_format=self.data_format)

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

    def get_cells_in_frame(self, frame):
        """Count the number of cells in the given frame.

        Args:
            frame (int): counts cells in this frame.

        Returns:
            list: All cell labels in the frame.
        """
        cells = np.unique(self._get_frame(self.y, frame))
        cells = np.delete(cells, np.where(cells == 0))  # remove the background
        return list(cells)

    def get_feature_shape(self, feature_name):
        """Return the shape of the requested feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            tuple: The shape of the feature.

        Raises:
            ValueError: feature_name is invalid.
        """
        channels = self.x.shape[self.channel_axis]

        # future area is just a neighborhood of a different frame.
        if feature_name == '~future area':
            feature_name = 'neighborhood'

        shape_dict = {
            'appearance': (self.crop_dim, self.crop_dim, channels),
            'neighborhood': (2 * self.neighborhood_scale_size + 1,
                             2 * self.neighborhood_scale_size + 1,
                             channels),
            'regionprop': (3,),
            'distance': (2,),
        }
        try:
            shape = shape_dict[feature_name]
        except KeyError:
            raise ValueError('{} is an invalid feature name. '
                             'Use one of {}'.format(
                                 feature_name, shape_dict.keys()))
        # shift the channel axis (it is channels_last by default)
        if len(shape) > 1 and self.data_format == 'channels_first':
            shape = tuple([shape[-1]] + list(shape[:-1]))
        return shape

    def _create_new_track(self, frame, old_label):
        """
        This function creates new tracks
        """
        new_track = len(self.tracks)
        new_label = new_track + 1
        self.tracks[new_track] = {
            'label': new_label,
            'frames': [frame],
            'frame_labels': [old_label],
            'daughters': [],
            'capped': False,
            'frame_div': None,
            'parent': None,
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
        unique_cells = self.get_cells_in_frame(frame)
        self.frame_features = self.get_frame_features(frame, unique_cells)
        for cell_idx, cell_label in enumerate(unique_cells):
            self._create_new_track(frame, cell_label)
            track_id = max(self.tracks)  # newly added track
            for f in self.frame_features:
                self.tracks[track_id][f] = self.frame_features[f][[cell_idx]]

        # Start a tracked label array
        self.y_tracked = self.y[[frame]].astype('int32')

    def compute_distance(self, track_centroids, frame_centroids):
        """Computes the distance between two centroids.

        Args:
            track_centroids (tuple): x and y centroid for the given track.
            frame_centroids (tuple): x and y centroid for the given frame.

        Returns:
            tuple: the distances for tracks and frames,
                and a boolean indicating whether the distance is valid.
        """
        centroids = np.concatenate([
            track_centroids,
            np.array([frame_centroids])
        ], axis=0)

        distances = np.concatenate([
            np.zeros((1, 2), dtype=self.dtype),
            np.diff(centroids, axis=0)
        ], axis=0)

        l2 = np.linalg.norm(distances, axis=0)
        is_cell_in_range = np.all(l2 <= self.max_distance)

        return distances[0:-1, :], distances[-1, :], is_cell_in_range

    def _fetch_tracked_feature(self, tracks_with_frames, feature):
        """Get feature data from each tracked frame less than before_frame.

        Args:
            tracks_with_frames (list): List of tuples,
                nodes and tracks to fetch.
            feature (str): Name of feature to fetch from tracked data.

        Returns:
            dict: dictionary of feature name to np.array of feature data.
        """
        feature_shape = self.get_feature_shape(feature)
        batches = len(tracks_with_frames)
        if self.data_format == 'channels_first':
            shape = tuple([batches, feature_shape[0], self.track_length] +
                          list(feature_shape)[1:])
        else:
            shape = tuple([batches, self.track_length] + list(feature_shape))

        tracked_feature = np.zeros(shape, dtype=self.dtype)
        for i, (n, valid_frames) in enumerate(tracks_with_frames):
            frame_dict = {frame: i for i, frame in enumerate(valid_frames)}
            frames = valid_frames[-self.track_length:]

            if len(frames) != self.track_length:
                # Pad the the frames with the last frame if not enough
                num_missing = self.track_length - len(frames)
                frames = frames + [frames[-1]] * num_missing

            # Get the feature data from the identified frames
            fetched = self.tracks[n][feature][[frame_dict[f] for f in frames]]
            tracked_feature[i] = fetched
        return tracked_feature

    def fetch_tracked_features(self, before_frame=None):
        """Get all feature data from each tracked frame less than before_frame.

        Args:
            before_frame (int, optional): The maximum frame to from which to
                fetch feature data.

        Returns:
            dict: dictionary of feature name to np.array of feature data.
        """
        t = timeit.default_timer()

        if before_frame is None:
            before_frame = self.x.shape[self.time_axis] + 1  # all frames

        track_valid_frames = ((n, d['frames'][:before_frame])
                              for n, d in self.tracks.items())
        tracks_with_frames = [(n, f) for n, f in track_valid_frames if f]

        tracked_features = {}
        for feature in self.features:
            fetched = self._fetch_tracked_feature(tracks_with_frames, feature)
            tracked_features[feature] = fetched

        self.logger.debug('Fetched tracked features in %s s.',
                          timeit.default_timer() - t)
        return tracked_features

    def get_frame_features(self, frame, cells_in_frame):
        """Get all features for each cell in the given frame.

        Args:
            frame (int): the frame number to calculate features.
            cells_in_frame (list): cell_labels in the frame.

        Returns:
            dict: dictionary of feature names to feature data
                for each cell in the frame.
        """
        t = timeit.default_timer()
        frame_features = {}
        all_features = list(self.features)
        if 'neighborhood' in self.features:
            all_features.append('~future area')
        for feature in all_features:
            feature_shape = self.get_feature_shape(feature)
            shape = tuple([len(cells_in_frame)] + list(feature_shape))
            frame_features[feature] = np.zeros(shape, dtype=self.dtype)
        # Fill frame_features with the proper values
        for cell_idx, cell_id in enumerate(sorted(cells_in_frame)):
            cell_features = self._get_features(frame, cell_id)
            for feature in cell_features:
                frame_features[feature][cell_idx] = cell_features[feature]
        self.logger.debug('Got all features for %s cells in frame %s in %s s.',
                          len(cells_in_frame), frame,
                          timeit.default_timer() - t)
        return frame_features

    def _get_input_pairs(self, frame):
        """Get all input pairs, inputs, and invalid pairs.

        Args:
            frame (int): Returns input pairs for only the given frame.

        Returns:
            list: input_pairs, list of tuples of frame and cell to use as input.
            dict: inputs for the given input pairs.
            list: invalid pairs of tracks and cells to ignore in assignment.
        """
        cells_in_frame = self.get_cells_in_frame(frame)

        # Get the features for previously tracked data
        track_features = self.fetch_tracked_features()

        # Get the features for the current frame
        self.frame_features = self.get_frame_features(frame, cells_in_frame)

        t = timeit.default_timer()  # don't time the other functions
        # Call model.predict only on inputs that are near each other
        inputs = {feature: ([], []) for feature in self.features}
        input_pairs, invalid_pairs = [], []

        # Fill the input matrices
        for track in range(len(self.tracks)):
            # capped tracks are not allowed to have assignments
            if self.tracks[track]['capped']:
                bad_pairs = [(track, c) for c in range(len(cells_in_frame))]
                invalid_pairs.extend(bad_pairs)
                continue

            # we need to get the future frame for the track we are comparing to
            try:
                frame_idx = self.tracks[track]['frames'].index(frame - 1)
                track_frame_features = {f: self.tracks[track][f][[frame_idx]]
                                        for f in self.frame_features}
            except ValueError:  # track may not exist in previous frame
                # if this happens, default to the cell's neighborhood
                track_frame_features = dict()

            for cell, _ in enumerate(cells_in_frame):
                feature_vals = {}

                # If distance is a feature it is used to exclude
                # impossible pairings from the get_feature call
                if 'distance' in self.features:
                    track_feature = track_features['distance'][track]
                    frame_feature = self.frame_features['distance'][cell]

                    track_feature, frame_feature, is_cell_in_range = \
                        self.compute_distance(track_feature, frame_feature)

                    # Set the distance feature
                    feature_vals['distance'] = (track_feature, frame_feature)
                else:
                    # not worried about distance, just calculate features
                    is_cell_in_range = True

                if not is_cell_in_range:
                    # Cell is outside of range, set cost to max and move on
                    invalid_pairs.append((track, cell))
                    continue

                # The cell is within range so we should add
                # all the information for all features
                for feature in self.features:
                    if feature == 'distance':
                        continue  # already calculated distance feature

                    track_feature = track_features[feature][track]
                    frame_feature = self.frame_features[feature][cell]

                    # this condition changes `frame_feature`
                    if feature == 'neighborhood':
                        if '~future area' in track_frame_features:
                            frame_feature = self._get_frame(
                                track_frame_features['~future area'], 0)

                    feature_vals[feature] = (track_feature, frame_feature)

                input_pairs.append((track, cell))
                for feature, (track_feature, frame_feature) in feature_vals.items():
                    inputs[feature][0].append(track_feature)
                    inputs[feature][1].append(frame_feature)

        for feature in self.features:
            in1, in2 = inputs[feature]
            feature_shape = self.get_feature_shape(feature)
            in1 = np.reshape(np.stack(in1),
                             tuple([len(input_pairs), self.track_length] +
                                   list(feature_shape)))
            in2 = np.reshape(np.stack(in2), tuple([len(input_pairs), 1] +
                                                  list(feature_shape)))
            inputs[feature] = (in1, in2)

        self.logger.debug('Got %s input pairs for frame %s in %s s.',
                          len(input_pairs), frame, timeit.default_timer() - t)
        return input_pairs, inputs, invalid_pairs

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
        cells_in_frame = self.get_cells_in_frame(frame)
        assignment_shape = (len(self.tracks), len(cells_in_frame))
        assignment_matrix = np.zeros(assignment_shape, dtype=self.dtype)

        input_pairs, inputs, invalid_pairs = self._get_input_pairs(frame)

        t = timeit.default_timer()
        if not input_pairs:  # frame is empty
            predictions = []
            assignment_matrix.fill(1)
        else:
            model_input = [ins for f in self.features for ins in inputs[f]]
            predictions = self.model.predict(model_input)
            assignment_matrix[tuple(zip(*input_pairs))] = 1 - predictions[:, 1]
            assignment_matrix[tuple(zip(*invalid_pairs))] = 1

        # Assemble full cost matrix
        cost_matrix = self._build_cost_matrix(assignment_matrix)
        self.logger.debug('Built cost matrix for frame %s in %s s.',
                          frame, timeit.default_timer() - t)
        return cost_matrix, dict(zip(input_pairs, predictions))

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
        cells_in_frame = self.get_cells_in_frame(frame)

        # Number of lables present in the current frame (needed to build cost matrix)
        y_tracked_update = np.zeros((1, self.y.shape[1], self.y.shape[2], 1),
                                    dtype='int32')

        for a in range(assignments.shape[0]):
            track, cell = assignments[a]

            # map the index from the LAP assignment to the cell label
            try:
                cell_id = cells_in_frame[cell]
            except IndexError:
                # cell has "died" or is a shadow assignment
                # no assignment should be made
                continue

            cell_features = {f: self.frame_features[f][[cell]]
                             for f in self.frame_features}

            if track in self.tracks:  # Add cell and frame to track
                self.tracks[track]['frames'].append(frame)
                self.tracks[track]['frame_labels'].append(cell_id)
                for feature in cell_features:
                    self.tracks[track][feature] = np.concatenate([
                        self.tracks[track][feature],
                        cell_features[feature],
                    ], axis=0)

                # Labels and indices differ by 1
                y_tracked_update[self.y[[frame]] == cell_id] = track + 1
                self.y[frame][self.y[frame] == cell_id] = track + 1

            else:  # Create a new track if there was a birth
                self._create_new_track(frame, cell_id)
                new_track_id = max(self.tracks)
                new_label = new_track_id + 1
                self.logger.info('Created new track for cell %s.', new_label)

                # Update features for new track from frame features
                self.tracks[new_track_id].update(cell_features)

                # See if the new track has a parent
                parent = self._get_parent(frame, cell, predictions)
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
            self._create_new_track(frame, self.tracks[track]['label'])

            for f in self.frame_features:
                self.tracks[new_track_id][f] = self.tracks[track][f][[frame_idx]]

            self.tracks[new_track_id]['parent'] = track

            # Remove features and frame from old track
            del self.tracks[track]['frames'][frame_idx]
            del self.tracks[track]['frame_labels'][frame_idx]
            for f in self.frame_features:
                self.tracks[track][f] = np.delete(
                    self.tracks[track][f], frame_idx, axis=0)
            self.tracks[track]['daughters'].append(new_track_id)

            # Change y_tracked_update
            y_tracked_update[self.y[[frame]] == new_label] = new_track_id + 1
            self.y[frame][self.y[frame] == new_label] = new_track_id + 1

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
        for (track, cell_id), p in predictions.items():
            prob = p[2]  # probability cell is part of the track
            # Make sure capped tracks can't be assigned parents
            if cell_id == cell and not self.tracks[track]['capped']:
                # Do not call a newly-appeared sibling of "cell" a parent
                if self.tracks[track]['frames'] == [frame]:
                    continue
                if prob > max_prob:
                    parent_id, max_prob = track, prob
        return parent_id

    def _sub_area(self, X_frame, y_frame, cell_label):
        """Fetch a neighborhood surrounding the cell in the given frame.

        Slices out a neighborhood_true_size square region around the center of
        the provided cell, and reshapes it to neighborhood_scale_size square.

        Args:
            X_frame (np.array): 2D numpy array, a frame of raw data.
            y_frame (np.array): 2D numpy array, a frame of annotated data.
            cell_label (int): The label of the cell to slice out.

        Returns:
            numpy.array: the resized region of X_frame around cell_label.
        """
        pads = ((self.neighborhood_true_size, self.neighborhood_true_size),
                (self.neighborhood_true_size, self.neighborhood_true_size),
                (0, 0))

        X_padded = np.pad(X_frame, pads, mode='constant', constant_values=0)
        y_padded = np.pad(y_frame, pads, mode='constant', constant_values=0)

        roi = (y_padded == cell_label).astype('int32')
        props = regionprops(np.squeeze(roi), coordinates='rc')

        center_x, center_y = props[0].centroid
        center_x, center_y = np.int(center_x), np.int(center_y)

        x1 = center_x - self.neighborhood_true_size
        x2 = center_x + self.neighborhood_true_size

        y1 = center_y - self.neighborhood_true_size
        y2 = center_y + self.neighborhood_true_size

        X_reduced = X_padded[x1:x2, y1:y2]

        # resize to neighborhood_scale_size
        resize_shape = (2 * self.neighborhood_scale_size + 1,
                        2 * self.neighborhood_scale_size + 1)
        X_reduced = resize(X_reduced, resize_shape,
                           data_format=self.data_format)

        return X_reduced

    def _get_features(self, frame, cell_label):
        """Gets the features of the cell in the frame.

        Args:
            frame (int): frame from which to get cell features.
            cell_label (int): label of the cell.

        Returns:
            dict: a dictionary with keys as the feature names.
        """
        # Get the bounding box
        X_frame = self._get_frame(self.x, frame)
        y_frame = self._get_frame(self.y, frame)

        roi = (y_frame == cell_label).astype('int32')
        props = regionprops(np.squeeze(roi), coordinates='rc')[0]

        centroid = props.centroid
        rprop = np.array([
            props.area,
            props.perimeter,
            props.eccentricity
        ])

        # Extract images from bounding boxes
        minr, minc, maxr, maxc = props.bbox
        if self.data_format == 'channels_first':
            appearance = np.copy(X_frame[:, minr:maxr, minc:maxc])
        else:
            appearance = np.copy(X_frame[minr:maxr, minc:maxc, :])

        # Resize images from bounding box
        appearance = resize(appearance, (self.crop_dim, self.crop_dim),
                            data_format=self.data_format)

        # Get the neighborhood
        neighborhood = self._sub_area(X_frame, y_frame, cell_label)

        # Try to assign future areas if future frame is available
        # TODO: We shouldn't grab a future frame if the frame is dark (was padded)
        try:
            X_future_frame = self._get_frame(self.x, frame + 1)
            future_area = self._sub_area(X_future_frame, y_frame, cell_label)
        except IndexError:
            future_area = neighborhood

        # future areas are not a feature instead a part of the neighborhood feature
        return {
            'appearance': np.expand_dims(appearance, axis=0),
            'distance': np.expand_dims(centroid, axis=0),
            'neighborhood': np.expand_dims(neighborhood, axis=0),
            'regionprop': np.expand_dims(rprop, axis=0),
            '~future area': np.expand_dims(future_area, axis=0)
        }

    def track_cells(self):
        """Tracks all of the cells in every frame.
        """
        start = timeit.default_timer()
        self._initialize_tracks()

        for frame in range(1, self.x.shape[self.time_axis]):
            t = timeit.default_timer()
            self.logger.info('Tracking frame %s', frame)

            cost_matrix, predictions = self._get_cost_matrix(frame)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignments = np.stack([row_ind, col_ind], axis=1)

            self._update_tracks(assignments, frame, predictions)
            self.logger.info('Tracked frame %s in %s s.',
                             frame, timeit.default_timer() - t)
        self.logger.info('Tracked all %s frames in %s s.',
                         self.x.shape[self.time_axis],
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
                'X': self.x,
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
            # Prep filepath
            filename = pathlib.Path(filename)
            if filename.suffix != '.trk':
                filename = filename.with_suffix('.trk')

            filename = str(filename)

            # Save
            with tarfile.open(filename, 'w') as trks:
                with tempfile.NamedTemporaryFile('w') as lineage_file:
                    json.dump(track_review_dict['tracks'], lineage_file, indent=1)
                    lineage_file.flush()
                    trks.add(lineage_file.name, 'lineage.json')

                with tempfile.NamedTemporaryFile() as raw_file:
                    np.save(raw_file, track_review_dict['X'])
                    raw_file.flush()
                    trks.add(raw_file.name, 'raw.npy')

                with tempfile.NamedTemporaryFile() as tracked_file:
                    np.save(tracked_file, track_review_dict['y_tracked'])
                    tracked_file.flush()
                    trks.add(tracked_file.name, 'tracked.npy')

        return track_review_dict

    def dump(self, filename):
        """Writes the state of the cell tracker to a .trk ('track') file.
        Includes raw & tracked images, and a lineage.json for parent/daughter
        information.
        """
        track_review_dict = self._track_review_dict()
        filename = pathlib.Path(filename)

        if filename.suffix != '.trk':
            filename = filename.with_suffix('.trk')

        filename = str(filename)

        with tarfile.open(filename, 'w') as trks:
            with tempfile.NamedTemporaryFile('w') as lineage_file:
                json.dump(track_review_dict['tracks'], lineage_file, indent=1)
                lineage_file.flush()
                trks.add(lineage_file.name, 'lineage.json')

            with tempfile.NamedTemporaryFile() as raw_file:
                np.save(raw_file, track_review_dict['X'])
                raw_file.flush()
                trks.add(raw_file.name, 'raw.npy')

            with tempfile.NamedTemporaryFile() as tracked_file:
                np.save(tracked_file, track_review_dict['y_tracked'])
                tracked_file.flush()
                trks.add(tracked_file.name, 'tracked.npy')

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
            fp_label = int(node.split('_')[0])
            fp_frame = int(node.split('_')[1])

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


cell_tracker = CellTracker  # allow backwards compatibility imports
