"""Tests for trk_io"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io

import numpy as np

import pytest

from deepcell_tracking import trk_io
from deepcell_tracking.test_utils import get_image


def test_save_trks(tmpdir):
    X = get_image(30, 30)
    y = np.random.randint(low=0, high=10, size=X.shape)
    lineage = [dict()]

    tempdir = str(tmpdir)
    with pytest.raises(ValueError):
        badfilename = os.path.join(tempdir, 'x.trk')
        trk_io.save_trks(badfilename, lineage, X, y)

    filename = os.path.join(tempdir, 'x.trks')
    trk_io.save_trks(filename, lineage, X, y)
    assert os.path.isfile(filename)

    # test saved tracks can be loaded
    loaded = trk_io.load_trks(filename)
    assert loaded['lineages'] == lineage
    np.testing.assert_array_equal(X, loaded['X'])
    np.testing.assert_array_equal(y, loaded['y'])

    # test save trks to bytes
    b = io.BytesIO()
    trk_io.save_trks(b, lineage, X, y)

    # load trks from bytes
    b.seek(0)
    loaded = trk_io.load_trks(b)
    assert loaded['lineages'] == lineage
    np.testing.assert_array_equal(X, loaded['X'])
    np.testing.assert_array_equal(y, loaded['y'])


def test_save_trk(tmpdir):
    X = get_image(30, 30)
    y = np.random.randint(low=0, high=10, size=X.shape)
    lineage = [dict()]

    tempdir = str(tmpdir)
    with pytest.raises(ValueError):
        badfilename = os.path.join(tempdir, 'x.trks')
        trk_io.save_trk(badfilename, lineage, X, y)

    with pytest.raises(ValueError):
        trk_io.save_trk('x.trk', [{}, {}], X, y)

    filename = os.path.join(tempdir, 'x.trk')
    trk_io.save_trk(filename, lineage, X, y)
    assert os.path.isfile(filename)

    # test saved tracks can be loaded
    loaded = trk_io.load_trks(filename)
    assert loaded['lineages'] == lineage
    np.testing.assert_array_equal(X, loaded['X'])
    np.testing.assert_array_equal(y, loaded['y'])

    # test save trks to bytes
    b = io.BytesIO()
    trk_io.save_trk(b, lineage, X, y)

    # load trks from bytes
    b.seek(0)
    loaded = trk_io.load_trks(b)
    assert loaded['lineages'] == lineage
    np.testing.assert_array_equal(X, loaded['X'])
    np.testing.assert_array_equal(y, loaded['y'])


def test_load_trks(tmpdir):
    filename = os.path.join(str(tmpdir), 'bad-lineage.trk')
    X = get_image(30, 30)
    y = np.random.randint(low=0, high=10, size=X.shape)
    lineage = [dict()]

    trk_io.save_track_data(filename=filename,
                           lineages=lineage,
                           raw=X,
                           tracked=y,
                           lineage_name='bad-lineage.json')

    with pytest.raises(ValueError):
        trk_io.load_trks(filename)
