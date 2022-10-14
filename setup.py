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
import os

from codecs import open
from setuptools import setup
from setuptools import find_packages


here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, 'README.md'), 'r', 'utf-8') as f:
    readme = f.read()


VERSION = '0.6.3'
NAME = 'DeepCell_Tracking'
DESCRIPTION = 'Tracking cells and lineage with deep learning.'
LICENSE = 'LICENSE'
AUTHOR = 'Van Valen Lab'
AUTHOR_EMAIL = 'vanvalenlab@gmail.com'
URL = 'https://github.com/vanvalenlab/deepcell-tracking'
DOWNLOAD_URL = ('https://github.com/vanvalenlab/'
                'deepcell-tracking/tarball/{}'.format(VERSION))


setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      install_requires=['networkx>=2.1',
                        'numpy',
                        'pandas',
                        'scipy',
                        'scikit-image>=0.14.5',
                        'deepcell-toolbox~=0.11.2'
                        ],
      extras_require={
          'tests': ['pytest<6',
                    'pytest-pep8',
                    'pytest-cov',
                    'pytest-mock']},
      long_description=readme,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9'])
