# ![DeepCell Tracking Banner](https://raw.githubusercontent.com/vanvalenlab/deepcell-tracking/master/docs/images/DeepCell_tracking_Banner.png)

[![PyPI version](https://badge.fury.io/py/DeepCell-Tracking.svg)](https://badge.fury.io/py/DeepCell-Tracking)
[![Build Status](https://github.com/vanvalenlab/deepcell-tracking/workflows/build/badge.svg)](https://github.com/vanvalenlab/deepcell-tracking/actions)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/deepcell-tracking/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/deepcell-tracking?branch=master)

`deepcell-tracking` uses deep learning models from [deepcell-tf](https://github.com/vanvalenlab/deepcell-tf) within an assignment problem framework to [track cells through time-lapse sequences](https://www.biorxiv.org/content/10.1101/803205v2) and build cell lineages. The assignment problem is solved using the [Hungarian algorithm.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2747604/)

## Getting Started

`deepcell-tracking` is a Python package that can be installed with `pip`:

```bash
pip install deepcell-tracking
```

Or it can be installed from source:

```bash
git clone https://github.com/vanvalenlab/deepcell-tracking.git

cd deepcell-tracking

# install the dependencies
pip install .
```

## How to Use

```python
from deepcell_tracking import CellTracker

# X and y are the time-sequence data and their corresponding segmentations (labels), respectively.
# model is a deepcell-tf tracking model.
tracker = CellTracker(X, y, model)

tracker.track_cells()  # runs in place, builds tracks

# Save all tracked data and lineage files to a .trk file
tracker.dump('./results.trk')

# Open the track file
from deepcell_tracking.utils import load_trks

data = load_trks('./results.trk')

lineage = data['lineages']  # linage information
X = data['X']  # raw X data
y = data['y']  # tracked y data
```
