import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from uapcar import dataset_iris
from uapcar import Glyph
from uapcar import plot_projection_2D
from uapcar import colors_contour_iris



# Example glyph:
dists_iris = dataset_iris()
glyph_iris = Glyph(dists_iris, 91, 181, True)
glyph_iris.save_off("glyph_iris", False)

# Example projection:
fig = plt.figure(figsize=(1., 1.))
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
plot_projection_2D(dists_iris, ax, bins=150, R=0.2, colors=colors_contour_iris)
plt.show()
