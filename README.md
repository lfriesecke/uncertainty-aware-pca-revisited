# Uncertainty-Aware PCA Revisited

This repository contains the official authors implementation associated with the IEEE VIS 2025 submission, "Uncertainty-Aware PCA Revisited".


## Overview

The codebase contains an interface for:
- computing and exporting a covariance stability glyph, that encodes the stability of the major eigenvectors of an uncertain covariance matrix
- projecting uncertain datasets to two dimensions, using a sampling-based approach


## Getting started

This code was tested using Python 1.13.


### Virtual environment:

Using a virtual environment for installing the necessary python dependencies is recommended. We used `virtualenv` for this:

1. Install `virtualenv` using `pip`

2. Navigate to the project folder and create a virtual environment named `venv_uapcar`:
   ```
   python -m venv venv_uapcar
   ```

3. Activate the virtual environment:
   
   ```
   source venv_uapcar/bin/activate      // on Linux & MacOS
   venv_uapcar/Scripts/activate.bat     // in Windows CMD
   venv_uapcar/Scripts/Activate.ps1     // in Windows Powershell
   ```

4. Install the required packages by running:

   ```
   pip install -r requirements.txt
   ```

5. Execute the given code - just use one of the provided datasets and export a glyph. If you want to reproduce the Iris glyph, run the `example.py` script.

6. Exit the virtual environment by running:

   ```
   deactivate
   ```


### Covariance stability glyph:

Computing a covariance stability glyph requires only a list of multivariate normal distributions. See the `MNDistribution` class in `distribution.py` for details. It is also possible to load the distributions used for the submission:

```python
from uapcar import dataset_iris

dists = dataset_iris()
```

The glyph can be computed by creating a new `Glyph` object and specifying the number of values for `alpha` ($\alpha \in [0, \pi]$) and `beta` ($\beta \in [0, 2\pi]$) for which the glyph should be evaluated. It is recommended to set `num_alpha` to `91` and `num_beta` to `181`:

```python
from uapcar import dataset_iris, Glyph

dists = dataset_iris()
glyph = Glyph(dists_iris, 91, 181)
```

Then, the computed glyph can then be saved to an OFF file using the `save_off` function:

```python
from uapcar import dataset_iris, Glyph

dists = dataset_iris()
glyph = Glyph(dists_iris, 91, 181)
glyph.save_off("glyph")
```


### Sampling based projection:

The `plot_projection_2D` function can be used to project a list of multivariate normal distributions. The function requires a list of distributions and an axis on which to display the final result. Also a list of colors for the contour lines can be specified. See `sampling.py` for other arguments and `utils/colors.py` for a list of colors used in the submission.

```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from uapcar import colors_contour_iris, dataset_iris, plot_projection_2D

dists = dataset_iris()

fig = plt.figure(figsize=(1., 1.))
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
plot_projection_2D(dists, ax, colors=colors_contour_iris)

plt.show()
```


## Datasets:

The following datasets were used for the submission:

1. **The Iris Dataset:**\
    Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.

2. **The Anuran Calls Dataset:**\
    Colonna, J., Nakamura, E., Cristo, M., & Gordo, M. (2015). Anuran Calls (MFCCs) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC9H.

3. **The Student Grades Dataset:**
    - Established by: T. Denoeux and M. . -H. Masson, "Principal component analysis of fuzzy data using autoassociative neural networks," in IEEE Transactions on Fuzzy Systems, vol. 12, no. 3, pp. 336-349, June 2004, doi: [10.1109/TFUZZ.2004.825990](https://doi.org/10.1109/TFUZZ.2004.825990).
    - Adopted as described in: J. Görtler, T. Spinner, D. Streeb, D. Weiskopf and O. Deussen, "Uncertainty-Aware Principal Component Analysis," in IEEE Transactions on Visualization and Computer Graphics, vol. 26, no. 1, pp. 822-831, Jan. 2020, doi: [10.1109/TVCG.2019.2934812](https://doi.org/10.1109/TVCG.2019.2934812).
