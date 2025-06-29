import numpy as np



def rot_2d(theta) -> np.ndarray:
    """Creates a 2D rotation matrix, representing a rotation according to the given angle."""

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
