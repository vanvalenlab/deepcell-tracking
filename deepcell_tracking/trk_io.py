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
"""Functions for reading and writing trk files"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import io
import json
import os
import re
import tarfile
import tempfile

import numpy as np


def load_trks(filename):
    """Load a trk/trks file.

    Args:
        filename (str or BytesIO): full path to the file including .trk/.trks
            or BytesIO object with trk file data

    Returns:
        dict: A dictionary with raw, tracked, and lineage data.
    """
    if isinstance(filename, io.BytesIO):
        kwargs = {'fileobj': filename}
    else:
        kwargs = {'name': filename}

    with tarfile.open(mode='r', **kwargs) as trks:

        # numpy can't read these from disk...
        with io.BytesIO() as array_file:
            array_file.write(trks.extractfile('raw.npy').read())
            array_file.seek(0)
            raw = np.load(array_file)

        with io.BytesIO() as array_file:
            array_file.write(trks.extractfile('tracked.npy').read())
            array_file.seek(0)
            tracked = np.load(array_file)

        # trks.extractfile opens a file in bytes mode, json can't use bytes.
        try:
            trk_data = trks.getmember('lineages.json')
        except KeyError:
            try:
                trk_data = trks.getmember('lineage.json')
            except KeyError:
                raise ValueError('Invalid .trk file, no lineage data found.')

        lineages = json.loads(trks.extractfile(trk_data).read().decode())
        lineages = lineages if isinstance(lineages, list) else [lineages]

        # JSON only allows strings as keys, so convert them back to ints
        for i, tracks in enumerate(lineages):
            lineages[i] = {int(k): v for k, v in tracks.items()}

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
    """Saves raw, tracked, and lineage data from multiple movies into one trks_file.

    Args:
        filename (str or io.BytesIO): full path to the final trk files or bytes object
            to save the data to
        lineages (list): a list of dictionaries saved as a json.
        raw (np.array): raw images data.
        tracked (np.array): annotated image data.

    Raises:
        ValueError: filename does not end in ".trks".
    """
    ext = os.path.splitext(str(filename))[-1]
    if not isinstance(filename, io.BytesIO) and ext != '.trks':
        raise ValueError('filename must end with `.trks`. Found %s' % filename)

    save_track_data(filename=filename,
                    lineages=lineages,
                    raw=raw,
                    tracked=tracked,
                    lineage_name='lineages.json')


def save_trk(filename, lineage, raw, tracked):
    """Saves raw, tracked, and lineage data for one movie into a trk_file.

    Args:
        filename (str or io.BytesIO): full path to the final trk files or bytes
            object to save the data to
        lineages (list or dict): a list of a single dictionary or a single
            lineage dictionary
        raw (np.array): raw images data.
        tracked (np.array): annotated image data.

    Raises:
        ValueError: filename does not end in ".trks".
    """
    ext = os.path.splitext(str(filename))[-1]
    if not isinstance(filename, io.BytesIO) and ext != '.trk':
        raise ValueError('filename must end with `.trk`. Found %s' % filename)

    # Check that lineages is a dictionary or list of length 1
    if isinstance(lineage, list):
        if len(lineage) > 1:
            raise ValueError('For trk file, lineages must be a dictionary '
                             'or list with a single dictionary')
        else:
            lineage = lineage[0]

    save_track_data(filename=filename,
                    lineages=lineage,
                    raw=raw,
                    tracked=tracked,
                    lineage_name='lineage.json')


def save_track_data(filename, lineages, raw, tracked, lineage_name):
    """Base function for saving tracking data as either trk or trks

    Args:
        filename (str or io.BytesIO): full path to the final trk files or bytes object
            to save the data to
        lineages (list or dict): a list of a single dictionary or a single lineage dictionarys
        raw (np.array): raw images data.
        tracked (np.array): annotated image data.
        lineage_name (str): Filename for the lineage file in the tarfile, either 'lineages.json'
            or 'lineage.json'
    """

    if isinstance(filename, io.BytesIO):
        kwargs = {'fileobj': filename}
    else:
        kwargs = {'name': filename}

    with tarfile.open(mode='w:gz', **kwargs) as trks:
        # disable auto deletion and close/delete manually
        # to resolve double-opening issue on Windows.
        with tempfile.NamedTemporaryFile('w', delete=False) as lineages_file:
            json.dump(lineages, lineages_file, indent=4)
            lineages_file.flush()
            lineages_file.close()
            trks.add(lineages_file.name, lineage_name)
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
