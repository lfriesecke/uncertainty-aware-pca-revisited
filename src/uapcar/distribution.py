import numpy as np

from dataclasses import dataclass
from .utils.linalg import rot_2d



@dataclass
class MNDistribution:
    """Defines multivariate normal distribution.

    Attributes
    ----------
    mean : np.ndarray
        Mean of specified distribution.
        
    cov : np.ndarray
        Covariance matrix of specified distribution.
        
    name : str
        Custom name for specified distribution.
    """

    mean: np.ndarray
    cov: np.ndarray
    name: str


    def __init__(self, mean: np.ndarray, cov: np.ndarray, name: str = "dist"):
        """Initializes distribution according to given mean and covariance matrix."""

        if len(mean) != cov.shape[0]:
            raise ValueError("Dimensionality of 'mean' and covariance matrix 'cov' do not match.")
        if not np.allclose(cov, cov.T):
            raise ValueError("Given covariance matrix 'cov' is not symmetric.")
        if not np.all(np.linalg.svd(cov)[1] >= 0):
            raise ValueError("Given covariance matrix 'cov' is not positive definite.")

        self.mean = mean
        self.cov = cov
        self.name = name


    def samples(self, num_samples: int, seed: int) -> np.ndarray:
        """Returns specified amount of samples according to distribution and given seed."""

        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.mean, self.cov, num_samples)


    def dim(self) -> int:
        """Returns dimensionality of distribution."""

        return len(self.mean)
    

    def translate(self, t: np.ndarray):
        """Returns new distribution, translated by the given vector t."""

        # Check if given translation vector has correct size:
        if len(t.shape) > 1:
            t = t.reshape(t.size)
        if t.size != self.dim():
            raise ValueError(f"Given translation vector has to have size: {self.dim()}.")
        
        # Create new translated distribution:
        return MNDistribution(self.mean + t, self.cov)


    def rotate(self, angle: float):
        """
        Returns new distribution, rotated by given angle; requires distribution to be
        2-dimensional.
        """

        # Assert distribution is two-dimensional:
        if self.dim() != 2:
            raise TypeError("Dimensionality of distribution has to be 2.")
        
        # Calc new mean and covariance matrix:
        R = rot_2d(angle)
        mean = R @ self.mean
        cov = R @ self.cov @ R.T

        # Create and return new distribution:
        return MNDistribution(mean, cov)


    def principal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcs principal axes of the distribution scaled by the corresponding eigenvector;
        requires distribution to be 2-dimensional.
        """

        # Assert distribution is two-dimensional:
        if self.dim() != 2:
            raise TypeError("Dimensionality of distribution has to be 2.")
        
        # Calc eigenvecs / eigenvals:
        U, S, _ = np.linalg.svd(self.cov)
        return (U[:,0] * S[0], U[:,1] * S[1])
    

    def angle_diameter_width(self) -> tuple[float, float, float]:
        """
        Returns angle between x-axis and diameter, as well as diameter and width of ellipse
        corresponding to bivariate normal distribution; requires distribution to be 
        2-dimensional.
        """

        # Assert distribution is two-dimensional:
        if self.dim() != 2:
            raise TypeError("Dimensionality of distribution has to be 2.")

        # Calc eigenvecs / eigenvals and enforce negative y value on diameter eigenvec:
        U, S, _ = np.linalg.svd(self.cov)
        if U[1, 0] >= 0:
            U[:,0] *= -1
        angle_rad = np.acos(U[:,0] @ np.array([1.0, 0.0]) / np.linalg.norm(U[:,0]))
        
        return (float(angle_rad), S[0]**0.5, S[1]**0.5)
    

    def main_axes_onto_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Projects the main axes of the distribution onto the given vector; requires distribution
        to be 2-dimensional.
        """

        # Assert distribution is two-dimensional:
        if self.dim() != 2:
            raise TypeError("Dimensionality of distribution has to be 2.")
        if v.size != self.dim():
            raise ValueError(f"Given translation vector has to have size: {self.dim()}.")
        
        # Normalize given vector:
        v = v / np.linalg.norm(v)

        # Calc principal axes:
        p = self.principal_axes()

        # Project both axes on given vector and return larger one:
        p1, p2 = [np.dot(p_, v) * v for p_ in p]
        return p1 if np.linalg.norm(p1) >= np.linalg.norm(p2) else p2
