import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from uapcar import dataset_anuran_calls
from uapcar import plot_projection_2D
from uapcar import colors_contour_anuran


dists_anuran = dataset_anuran_calls()
fig = plt.figure(figsize=(12., 12.))
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim((-1.5, 1.5))
ax.set_ylim((-1.5, 1.5))
plot_projection_2D(
    dists_anuran, 
    ax, 
    num_realizations = 1000000,
    bins = 250, 
    R = 0.2, 
    colors = colors_contour_anuran, 
    modify_ax = True, 
    display_progress = True
)
plt.savefig('projection_anuran_calls.pdf')
