import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

from uapcar import dataset_students_grades
from uapcar import plot_projection_2D
from uapcar import colors_contour_students


dists_students = dataset_students_grades()
fig = plt.figure(figsize=(12., 12.))
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim((-15.0, 15.0))
ax.set_ylim((-15.0, 15.0))
plot_projection_2D(
    dists_students, 
    ax, 
    num_realizations = 1000000,
    bins = 250, 
    R = 0.2, 
    colors = colors_contour_students, 
    scale_matrix = np.array([[-1., 0.], [0., 1.]]),
    modify_ax = True, 
    display_progress = True
)
plt.savefig('projection_student_grades.pdf')
