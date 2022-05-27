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
"""Tests for tracking.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd

import pytest

from deepcell_tracking import tracking
from deepcell_tracking import utils
from deepcell_tracking import trk_io
from deepcell_tracking.test_utils import get_annotated_movie


class DummyModel(object):  # pylint: disable=useless-object-inheritance

    def predict(self, data):
        # Grab a random value from the data dict and select batch dim
        if data:
            batches = 1
            frames = 1
            pred_shape = next(iter(data.values())).shape
            cells = pred_shape[1]
        else:
            batches = 0
            frames = 0
            cells = 0

        return np.random.random((batches, cells, cells, frames, 3))


class DummyEncoder(object):  # pylint: disable=useless-object-inheritance

    def predict(self, data):
        # Data should be of the shape (frames, cells, 32, 32, 1)
        # Where frames = number of images in the movie or X.shape[0]
        # and   cells = number of cells (unique IDs) in the movie
        # so grab a random value from the data dict and store correct dim
        if data:
            pred_shape = next(iter(data.values())).shape
            frames = pred_shape[0]  # track length
            cells = pred_shape[1]
        else:
            frames = 0
            cells = 0

        return [np.random.random((frames, cells, 64)),
                np.random.random((frames, cells, 2))]


class TestTracking(object):  # pylint: disable=useless-object-inheritance

    def test_simple(self):
        data_format = 'channels_last'
        frames = 3
        labels_per_frame = 5
        y = get_annotated_movie(img_size=256,
                                labels_per_frame=labels_per_frame,
                                frames=frames,
                                mov_type='sequential', seed=0,
                                data_format=data_format)
        x = np.random.random(y.shape)
        num_objects = len(np.unique(y)) - 1
        model = DummyModel()
        encoder = DummyEncoder()

        _ = tracking.CellTracker(x, y,
                                 tracking_model=model,
                                 neighborhood_encoder=encoder)

        # test data with bad rank
        with pytest.raises(ValueError):
            tracking.CellTracker(
                np.random.random((32, 32, 1)),
                np.random.randint(num_objects, size=(32, 32, 1)),
                tracking_model=model,
                neighborhood_encoder=encoder)

        # test mismatched x and y shape
        with pytest.raises(ValueError):
            tracking.CellTracker(
                np.random.random((3, 32, 32, 1)),
                np.random.randint(num_objects, size=(2, 32, 32, 1)),
                tracking_model=model,
                neighborhood_encoder=encoder)

        # test bad data_format
        with pytest.raises(ValueError):
            tracking.CellTracker(x, y,
                                 tracking_model=model,
                                 neighborhood_encoder=encoder,
                                 data_format='invalid')

    def test_track_cells(self, tmpdir):
        frames = 10
        track_length = 3
        labels_per_frame = 3

        # TODO: test detected divisions
        # TODO: test creating new track

        # TODO: Fix for channels_first
        for data_format in ('channels_last',):  # 'channels_first'):

            y1 = get_annotated_movie(img_size=256,
                                     labels_per_frame=labels_per_frame,
                                     frames=frames,
                                     mov_type='sequential', seed=1,
                                     data_format=data_format)
            y2 = get_annotated_movie(img_size=256,
                                     labels_per_frame=labels_per_frame * 2,
                                     frames=frames,
                                     mov_type='sequential', seed=2,
                                     data_format=data_format)
            y3 = get_annotated_movie(img_size=256,
                                     labels_per_frame=labels_per_frame,
                                     frames=frames,
                                     mov_type='sequential', seed=3,
                                     data_format=data_format)

            y = np.concatenate((y1, y2, y3))

            x = np.random.random(y.shape)

            tracker = tracking.CellTracker(
                x, y,
                tracking_model=DummyModel(),
                neighborhood_encoder=DummyEncoder(),
                track_length=track_length,
                data_format=data_format)

            tracker.track_cells()

            # test tracker.dataframe
            df = tracker.dataframe(cell_type='test-value')
            assert isinstance(df, pd.DataFrame)
            assert 'cell_type' in df.columns  # pylint: disable=E1135

            # test incorrect values in tracker.dataframe
            with pytest.raises(ValueError):
                tracker.dataframe(bad_value=-1)

            # test tracker.postprocess
            tempdir = str(tmpdir)
            path = os.path.join(tempdir, 'postprocess.xyz')
            tracker.postprocess(filename=path)
            post_saved_path = os.path.join(tempdir, 'postprocess.trk')
            assert os.path.isfile(post_saved_path)

            # test tracker.dump
            path = os.path.join(tempdir, 'test.xyz')
            tracker.dump(path)
            dump_saved_path = os.path.join(tempdir, 'test.trk')
            assert os.path.isfile(dump_saved_path)

            # utility tests for loading trk files
            # TODO: move utility tests into utils_test.py

            # test trk_folder_to_trks
            trk_io.trk_folder_to_trks(tempdir, os.path.join(tempdir, 'all.trks'))
            assert os.path.isfile(os.path.join(tempdir, 'all.trks'))

            # test load_trks
            data = trk_io.load_trks(post_saved_path)
            assert isinstance(data['lineages'], list)
            assert all(isinstance(d, dict) for d in data['lineages'])
            np.testing.assert_equal(data['X'], tracker.X)
            np.testing.assert_equal(data['y'], tracker.y_tracked)
            # load trks instead of trk
            data = trk_io.load_trks(os.path.join(tempdir, 'all.trks'))

            # test trks_stats
            utils.trks_stats(os.path.join(tempdir, 'all.trks'))
