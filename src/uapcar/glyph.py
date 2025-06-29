import numpy as np
import os.path

from dataclasses import dataclass
from typing import Tuple

from .distribution import MNDistribution
from .utils.mandel import calc_mandel_T_d, to_mandel_notation
from .utils.output import print_progress
from .utils.pca import calc_m_overline, calc_C_overline, calc_C_m, calc_uncertain_global_covariance



@dataclass
class Glyph:
    """Defines a covariance stability glyph.

    Attributes
    ----------
    dists : list[MNDistribution]       
        List of multivariate normal distribution to compute glyph with.
    
    num_alpha : int
        Number of samples for α ∈ [0,π]. Will be set to at least 3 automatically.

    num_beta : int
        Number of samples for β ∈ [0,2π]. Will be set to at least 5 automatically.

    progress : bool
        Enables output of progress bars.
    """

    dists: list[MNDistribution]
    num_alpha: int
    num_beta: int
    progress: bool

    _num_samples: int
    _current_sample: int

    _vertices: np.ndarray
    _faces: np.ndarray
    _seed: int = 540839781


    def __init__(self, dists: list[MNDistribution], num_alpha: int, num_beta: int, progress: bool = False):
        """Initializes glyph object."""

        # Check if dists were given:
        if len(dists) == 0:
            raise ValueError("Given list of distributions is empty, but has to contain at least one distribution.")
        
        # Check if dists all have same number of dimensions:
        d = dists[0].dim()
        for dist in dists[1:]:
            if d != dist.dim():
                raise ValueError("Given distributions have different numbers of dimensions.")
        
        # Check if number of dimensions is at least 3:
        if d <= 2:
            raise ValueError("Given distributions have to have at least 3 dimensions to calculate a glyph.")

        # Init properties:
        self.dists = dists
        self.num_alpha = max(num_alpha, 3)
        self.num_beta = max(num_beta, 5)
        self.progress = progress

        self._num_samples = self.num_alpha * self.num_beta
        self._current_sample = 0

        # Calc glyph:
        np.random.seed(self._seed)
        self._process_glyph()
    

    def _process_glyph(self) -> None:
        """Samples the glyph using the given number of alpha and beta samples."""

        # Calc vertices:
        alpha = np.linspace(0.0, 2*np.pi, self.num_alpha, endpoint=True)
        beta = np.linspace(0.0, np.pi, self.num_beta, endpoint=True)
        self._vertices = np.array(
            [[[np.cos(a)*np.sin(b), np.sin(a)*np.sin(b), np.cos(b)] for a in alpha] for b in beta]
        ).reshape((self.num_alpha*self.num_beta), 3)

        # Calc global covariance matrix:
        N = len(self.dists)
        m_overline = calc_m_overline(self.dists)
        C_overline = calc_C_overline(self.dists)
        C_m = calc_C_m(self.dists, m_overline)
        C = C_m + (N-1)/N * C_overline

        # Calc major eigenvecs:
        U, _, _ = np.linalg.svd(C)
        U = U[:,:3]
        # idx = np.array([2, 1, 0]) # TODO: ask if wanted
        # U = U[:,idx]

        # Calc uncertain global mean and covariance:
        N_r = calc_uncertain_global_covariance(self.dists)

        # Evaluate spherical function:
        vals = np.array([self._val(N_r.cov, N_r.mean, U@v_) for (i, v_) in enumerate(self._vertices)])
        max_val = np.max(vals)
        self._vertices = ((self._vertices * vals[:, np.newaxis]) * 1/max_val)

        # Print progress:
        if self.progress:
            print_progress(100, 100)
            print()

        # Calc faces:
        self._faces = np.zeros(((self.num_alpha-1)*(self.num_beta-1)*2, 3), dtype=int)
        idx = 0
        for i, _ in enumerate(beta[:-1]):
            for j, _ in enumerate(alpha[:-1]):
                self._faces[idx]   = [i*self.num_alpha + j, (i+1)*self.num_alpha + j+1, i*self.num_alpha + j+1]
                self._faces[idx+1] = [i*self.num_alpha + j, (i+1)*self.num_alpha + j, (i+1)*self.num_alpha + j+1]
                idx += 2


    def _val(self, C: np.ndarray, m: np.ndarray, x: np.ndarray) -> float:
        """Calcs the value of the glyph for the given vector."""
        
        # Print progress:
        if self.progress:
            if self._current_sample % 10 == 0:
                print_progress(self._current_sample, self._num_samples)
            self._current_sample += 1

        # Do calculations:
        _, K = self._evsubspace(x)
        return self._integralc(C, m, K)


    def _evsubspace(self, x: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
        """Calcs subspaces B and K using Gram-Schmidt algorithm"""

        # Initialize values:
        n = len(x)
        r = (n * (n + 1)) // 2
        B = np.zeros((r, n))
        u = x / np.linalg.norm(x)

        # Gram-Schmidt algorithm:
        def GS(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            
            # Initialize with random matrix (and u):
            S = np.random.rand(n, r)
            S[:,0] = u

            # Apply Gram-Schmidt algorithm:
            for i in range(1, r):
                v = S[:,i] - np.dot(S[:,i], u) * u
                S[:,i] = v / np.linalg.norm(v)
            
            # Convert to matrix of mandel vectors:
            Sc = np.zeros((r, r))
            T, d = calc_mandel_T_d(n)
            for i in range(0, r):
                Sc[:,i] = to_mandel_notation(np.outer(S[:,i], S[:,i]), T, d)
            
            # Return SVD of matrix of mandel vectors:
            return np.linalg.svd(Sc)
        
        # Repeat Gram-Schmidt algorithm until valid solutions is found:
        rnk = -1
        U = np.zeros((r, r))
        while rnk != (r - n + 1):
            U, S, _ = GS(x)
            rnk = np.sum(S > S[0] * eps)

        return (U[:, 0:rnk], U[:, rnk:])
    

    def _integralc(self, C: np.ndarray, m: np.ndarray, K: np.ndarray) -> float:
        """Calcs the integral over the n-dimensional linear subspace."""

        _, k = K.shape
        x0 = self._argminsubspace(C, m, K)
        s_x0 = self._mahalanobis(C, m, x0)

        return 1/np.sqrt(np.pow(2*np.pi, k) * np.linalg.det(K.T @ C @ K)) * np.exp(-s_x0/2)
    

    def _argminsubspace(self, C: np.ndarray, m: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Calcs the argmin of the subspace."""
        
        invC = np.linalg.inv(C)
        n, k = K.shape
        M = np.block([[invC, K], [K.T, np.zeros((k, k))]])
        y = np.linalg.solve(M, np.concatenate([invC @ m, np.zeros(k)]))

        return y[:n]


    def _mahalanobis(self, C: np.ndarray, m: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calcs the squared Mahalanobis distance."""

        z = x - m
        return z.T @ np.linalg.inv(C) @ z


    def save_off(self, file_path: str = "glyph", check_file: bool = False) -> None:
        """Saves calculated glyph object to .off file.

        Parameters
        ----------
        file_path : str, optional
            Path of the resulting file. '.off' suffix will be added automatically. Defaults to 'glyph'.
        
        check_file : bool, optional
            If 'True' an exception will be thrown if a file at the given path already exists. Defaults to False.
        """

        # Check if file exists:
        file_path = file_path + ".off"
        if check_file and os.path.isfile(file_path):
            raise FileExistsError("A file at the given path already exists.")
        

        # Write content to file:
        with open(file_path, 'w') as file:

            # Add header:
            file.write("OFF\n")
            nv, _ = self._vertices.shape
            nf, _ = self._faces.shape
            file.write(f"{nv} {nf} 0\n")

            # Add vertices:
            for v_ in self._vertices:
                file.write(f"{v_[0]} {v_[1]} {v_[2]}\n")

            # Add faces:
            for f_ in self._faces:
                file.write(f"3 {f_[0]} {f_[1]} {f_[2]}\n")
