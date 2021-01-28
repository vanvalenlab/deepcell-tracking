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
"""Tests for tracking.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import skimage as sk

import pytest

from deepcell_tracking import tracking
from deepcell_tracking import utils


def _get_dummy_tracking_data(length=128, frames=3,
                             data_format='channels_last'):
    if data_format == 'channels_last':
        channel_axis = -1
    else:
        channel_axis = 0

    x, y = [], []
    while len(x) < frames:
        _x = sk.data.binary_blobs(length=length, n_dim=2)
        _y = sk.measure.label(_x)
        if len(np.unique(_y)) > 3:
            x.append(_x)
            y.append(_y)

    x = np.stack(x, axis=0)  # expand to 3D
    y = np.stack(y, axis=0)  # expand to 3D

    x = np.expand_dims(x, axis=channel_axis)
    y = np.expand_dims(y, axis=channel_axis)

    return x.astype('float32'), y.astype('int32')


class DummyModel(object):  # pylint: disable=useless-object-inheritance

    def predict(self, data):
        # Grab a random value from the data dict and select batch dim
        batches = 0 if not data else next(iter(data.values())).shape[0]

        return np.random.random((batches, 3))


class TestTracking(object):  # pylint: disable=useless-object-inheritance

    def test_simple(self):
        length = 128
        frames = 3
        x, y = _get_dummy_tracking_data(length, frames=frames)
        num_objects = len(np.unique(y)) - 1
        model = DummyModel()

        _ = tracking.CellTracker(x, y, model=model)

        # test data with bad rank
        with pytest.raises(ValueError):
            tracking.CellTracker(
                np.random.random((32, 32, 1)),
                np.random.randint(num_objects, size=(32, 32, 1)),
                model=model)

        # test mismatched x and y shape
        with pytest.raises(ValueError):
            tracking.CellTracker(
                np.random.random((3, 32, 32, 1)),
                np.random.randint(num_objects, size=(2, 32, 32, 1)),
                model=model)

        # test bad features
        with pytest.raises(ValueError):
            tracking.CellTracker(x, y, model=model, features=None)

        # test bad data_format
        with pytest.raises(ValueError):
            tracking.CellTracker(x, y, model=model, data_format='invalid')

    def test_get_feature_shape(self):
        length = 128
        frames = 3
        x, y = _get_dummy_tracking_data(length, frames=frames)
        model = DummyModel()

        for data_format in ('channels_first', 'channels_last'):
            tracker = tracking.CellTracker(x, y, model=model,
                                           data_format=data_format)

            for f in tracker.features:
                shape = tracker.get_feature_shape(f)
                assert isinstance(shape, tuple)

            with pytest.raises(ValueError):
                tracker.get_feature_shape('bad feature name')

    def test_track_cells(self):
        length = 128
        frames = 5
        track_length = 2

        features = ['appearance', 'neighborhood', 'regionprop', 'distance']

        # TODO: Fix for channels_first
        for data_format in ('channels_last',):  # 'channels_first'):

            x, y = _get_dummy_tracking_data(
                length, frames=frames, data_format=data_format)

            tracker = tracking.CellTracker(
                x, y,
                model=DummyModel(),
                track_length=track_length,
                data_format=data_format,
                features=features)

            tracker.track_cells()

            # test tracker.dataframe
            df = tracker.dataframe(cell_type='test-value')
            assert isinstance(df, pd.DataFrame)
            assert 'cell_type' in df.columns  # pylint: disable=E1135

            # test incorrect values in tracker.dataframe
            with pytest.raises(ValueError):
                tracker.dataframe(bad_value=-1)

            try:
                # test tracker.postprocess
                tempdir = tempfile.mkdtemp()  # create dir
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
                utils.trk_folder_to_trks(tempdir, os.path.join(tempdir, 'all.trks'))
                assert os.path.isfile(os.path.join(tempdir, 'all.trks'))

                # test load_trks
                data = utils.load_trks(post_saved_path)
                assert isinstance(data['lineages'], list)
                assert all(isinstance(d, dict) for d in data['lineages'])
                np.testing.assert_equal(data['X'], tracker.x)
                np.testing.assert_equal(data['y'], tracker.y_tracked)
                # load trks instead of trk
                data = utils.load_trks(os.path.join(tempdir, 'all.trks'))

                # test trks_stats
                utils.trks_stats(os.path.join(tempdir, 'test.trk'))
            finally:
                try:
                    shutil.rmtree(tempdir)  # delete directory
                except OSError as exc:
                    if exc.errno != errno.ENOENT:  # no such file or directory
                        raise  # re-raise exception

    def test_fetch_tracked_features(self):
        length = 128
        frames = 5

        # TODO: Fix for channels_first
        for data_format in ('channels_last',):  # 'channels_first'):

            x, y = _get_dummy_tracking_data(
                length, frames=frames, data_format=data_format)

            for track_length in (1, frames // 2 + 1, frames + 1):
                tracker = tracking.CellTracker(
                    x, y,
                    model=DummyModel(),
                    track_length=track_length,
                    data_format=data_format)

                tracker._initialize_tracks()

                axis = tracker.channel_axis

                tracked_features = tracker.fetch_tracked_features()

                for feature_name in tracker.features:
                    feature_shape = tracker.get_feature_shape(feature_name)
                    feature = tracked_features[feature_name]

                    axis = tracker.channel_axis
                    assert feature.shape[axis] == feature_shape[axis]
                    assert feature.shape[0] == len(tracker.tracks)
                    assert feature.shape[tracker.time_axis + 1] == track_length

                tracked_features = tracker.fetch_tracked_features(
                    before_frame=frames // 2 + 1)

                for feature_name in tracker.features:
                    feature_shape = tracker.get_feature_shape(feature_name)
                    feature = tracked_features[feature_name]

                    axis = tracker.channel_axis
                    assert feature.shape[axis] == feature_shape[axis]
                    assert feature.shape[0] == len(tracker.tracks)
                    assert feature.shape[tracker.time_axis + 1] == track_length

    def test__sub_area(self):
        length = 128
        frames = 3
        model = DummyModel()

        # TODO: Fix for channels_first
        for data_format in ('channels_last',):  # 'channels_first'):
            x, y = _get_dummy_tracking_data(
                length, frames=frames, data_format=data_format)

            tracker = tracking.CellTracker(
                x, y, model=model, data_format=data_format)

            for f in range(frames):
                if data_format == 'channels_first':
                    xf = x[:, f]
                    yf = y[:, f]
                else:
                    xf = x[f]
                    yf = y[f]

                    sub = tracker._sub_area(xf, yf, 1)
                    expected_shape = tracker.get_feature_shape('neighborhood')
                    assert sub.shape == expected_shape
