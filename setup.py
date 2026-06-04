import os

from codecs import open
from setuptools import setup
from setuptools import find_packages


here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, 'README.md'), 'r', 'utf-8') as f:
    readme = f.read()


VERSION = '0.6.5'
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
                        'numpy<2',
                        'pandas',
                        'scipy',
                        'scikit-image>=0.14.5',
                        'deepcell-toolbox~=0.12.0'
                        ],
      extras_require={
          'tests': ['pytest',
                    'pytest-cov',
                    'pytest-mock',
                    'ruff']},
      long_description=readme,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      python_requires='>=3.7, <3.11',
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10'])
