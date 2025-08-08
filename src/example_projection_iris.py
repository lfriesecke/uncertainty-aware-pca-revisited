import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

from uapcar import dataset_iris
from uapcar import plot_projection_2D
from uapcar import colors_contour_iris


dists_iris = dataset_iris()
fig = plt.figure(figsize=(12., 12.))
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim((-4., 4.))
ax.set_ylim((-4., 4.))
plot_projection_2D(
    dists_iris, 
    ax, 
    num_realizations = 1000000,
    bins = 250, 
    R = 0.2, 
    colors = colors_contour_iris, 
    scale_matrix = np.array([[-1., 0.], [0., 2.]]),
    modify_ax = True, 
    display_progress = True
)
plt.savefig('projection_iris.pdf')
