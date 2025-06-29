import matplotlib.axes as axes
import numpy as np

from .distribution import MNDistribution
from .utils.colors import colors_contour_default
from .utils.output import print_progress
from .utils.pca import align_direction_eigenvecs, projection_onto_subspace
from .utils.sampling import samples_from_distributions



def _modify_ax(
        ax: axes.Axes, 
        x_axis: bool, 
        y_axis: bool
) -> axes.Axes:
    """Modifies the look of the given axis to be more subtle."""
    
    # Define constants:
    colors_cosys = [0.3, 0.3, 0.3, 1.0]
    linestyle = (0, (1, 5))
    colors_borders = (0.5, 0.5, 0.5, 1.0)
    lw_borders = 1.5

    # Set aspect ratio and disable ticks:
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot x-axis:
    if x_axis:
        ax.plot([0.05, ax.get_xlim()[1]], [0,0], linestyle=linestyle, color=colors_cosys, zorder=0)
        ax.plot([-0.05, ax.get_xlim()[0]], [0,0], linestyle=linestyle, color=colors_cosys, zorder=0)
    
    # Plot y-axis:
    if y_axis:
        ax.plot([0,0], [0.05, ax.get_ylim()[1]], linestyle=linestyle, color=colors_cosys, zorder=0)
        ax.plot([0,0], [-0.05, ax.get_ylim()[0]], linestyle=linestyle, color=colors_cosys, zorder=0)
    
    # Modify borders:
    for spine in ax.spines.values():
        spine.set_edgecolor(colors_borders)
        spine.set_linewidth(lw_borders)
    return ax


def _radial_hann_window(
    c: np.ndarray, 
    R: float,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Defines the radial Hann window function."""
    
    cx, cy = x - c[0], y - c[1]
    dist_sqr = np.square(cx) + np.square(cy)
    dist = np.sqrt(dist_sqr)

    hann = (2*np.pi) / (np.square(R) * (np.square(np.pi) - 4)) * np.square(np.cos((np.pi * dist) / (2*R)))
    hann[dist >= R] = 0.0
    return hann


def _plot_contour_line(
    ax: axes.Axes,
    samples: np.ndarray,
    R: float,
    bins: int,
    levels: list[float],
    color: tuple[float, float, float, float],
    display_progress: bool,
) -> axes.Axes:
    """Plots the contour line for one set of points."""
    
    # Calc range of height map:
    x_min, x_max = np.min(samples[:,0]) - R, np.max(samples[:,0]) + R
    y_min, y_max = np.min(samples[:,1]) - R, np.max(samples[:,1]) + R

    # Calc X, Y and initialize Z for height field:
    x = np.linspace(x_min, x_max, bins)
    y = np.linspace(y_min, y_max, bins)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    # Calc height map:
    for i, s in enumerate(samples):
        Z += _radial_hann_window(s, R, X, Y)
        if display_progress and i % 10 == 0:
            print_progress(i, samples.shape[0])
    if display_progress:
        print_progress(100, 100)
        print()
    
    # Choose contour levels:
    vals = np.sort(Z.flatten())[::-1]
    V = np.sum(vals)

    abs_levels = []
    for rel_level in levels:
        v_ = V * rel_level
        sum = 0
        idx = 0
        while sum < v_:
            sum += vals[idx]
            idx += 1
        abs_levels.append((vals[idx - 1] + vals[idx]) / 2)
    
    # Plot contour lines:
    ax.contour(X, Y, Z, abs_levels, colors = color, linewidths = 1.0)
    return ax


def _plot_contour_lines(
    ax: axes.Axes,   
    samples: list[np.ndarray],
    R: float,
    bins: int,
    levels: list[float],
    colors: list[tuple[float, float, float, float]],
    display_progress: bool,
) -> axes.Axes:
    """Plots the contour lines for each distribution separately."""
    
    # Plot contour line for each distribution separately:
    for i, ls in enumerate(samples):
        _plot_contour_line(ax, ls, R, bins, levels, colors[i], display_progress)
    return ax



def plot_projection_2D(
    dists: list[MNDistribution],
    ax: axes.Axes,
    num_realizations: int = 10000,
    bins: int = 100,
    R: float = 0.25,
    levels: list[float] = [0.97, 0.78, 0.30],
    colors: list[tuple[float, float, float, float]] = colors_contour_default,
    scale_matrix: np.ndarray = np.identity(2),
    seed: int = 849273076560029784,
    modify_ax: bool = False,
    display_progress: bool = False,
) -> axes.Axes:
    """Plots the 2D projection of the given distribution by sampling the distributions and projecting
    each realization separately using PCA. Plots the projection by generating a height field using a 
    2D radial Hann function and drawing contour lines.

    Parameters
    ----------
    dists : list[MNDistribution]
        List of distributions.

    ax : axes.Axes
        Axes to add contour lines to.
    
    num_realizations : int, optional
        Number of realizations. Defaults to 10000.

    bins : int, optional
        Number of bins in each direction used for the final height field. Defaults to 100.

    R : float, optional
        Radius of the 2d radial hann function. Defaults to 0.25.
    
    levels : list[float], optional
        Levels of the contour lines. E.g. a level of 0.97 means 97 % of points lie within the contour line. 
        Defaults to [0.97, 0.78, 0.30].

    colors : list[tuple[float, float, float, float]], optional
        Colors of the contour lines for each distribution. Defaults to the colors specified by `colors_contour_default`.
    
    scale_matrix : np.ndarray, optional
        Additional scaling matrix, which can be used to manipulate the projected 2D points, e.g. by inverting the x 
        position. Defaults to the identity matrix.
    
    seed : int, optional
        Seed for generating realizations. Defaults to 849273076560029784.

    modify_ax : bool, optional
        Modifies given ax to look more like in the paper. Defaults to False.

    display_progress : bool, optional
        Enables progress bar. Defaults to False.    

    Returns
    -------
    ax : axes.Axes
        Given axes with contour lines.
    """

    # Sample distributions:
    sample_list, sample_matrix = samples_from_distributions(dists, num_realizations, seed)

    # Calc mean and cov and center points for each realization respectively:
    sample_matrix_R = sample_matrix.reshape(num_realizations, len(dists), dists[0].dim(), order='F')
    m_R = np.mean(sample_matrix_R, axis=1)
    sample_matrix_Rc = sample_matrix_R - m_R[:, np.newaxis, :]
    sample_list_c = [sl - m_R for sl in sample_list]
    C_R = np.stack([sample_matrix_Rc[i,:,:].T @ sample_matrix_Rc[i,:,:] / sample_matrix_Rc.shape[1] for i in range(sample_matrix_Rc.shape[0])])

    # Calc eigenvecs for each realization respectively:
    eigenvecs_R = [np.linalg.svd(c)[0] for c in C_R]
    eigenvecs_R = align_direction_eigenvecs(eigenvecs_R)
    sample_list_proj = [np.stack([projection_onto_subspace(eigenvecs_R[i][:,:2], sl[i]) for i in range(num_realizations)]) for sl in sample_list_c]
    sample_list_proj = [s @ scale_matrix for s in sample_list_proj]

    # Extend color list in case not enough colors were given:
    if len(colors) < len(dists):
        colors = [colors[i % len(colors)] for i in range(len(dists))]

    # Plot projection:
    ax = _plot_contour_lines(ax, sample_list_proj, R, bins, levels, colors, display_progress)

    # Modify axes:
    if modify_ax:
        ax = _modify_ax(ax, True, True)
    return ax
