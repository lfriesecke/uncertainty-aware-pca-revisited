import numpy as np

from ..distribution import MNDistribution
from ..utils.mandel import *



""" ==================================================
        Mean & Covariance
    ================================================== """

def calc_C(P: np.ndarray) -> np.ndarray:
    """Calcs the local covariance matrix of a set of n-dimensional points, given by:
    $C = \frac{1}{N} \sum_{i=1}{N}p_i p_i^T - m m^T$

    Parameters
    ----------
    P : np.ndarray
        Column-wise set of n-dimensional points.

    Returns
    -------
    C: np.ndarray
        `n*n`-dimensional covariance matrix.
    """

    mean = np.mean(P, axis=0).reshape(P.shape[1], 1)
    return 1/P.shape[0] * P.T @ P - mean @ mean.T


def calc_m_overline(dists: list[MNDistribution]) -> np.ndarray:
    """Calcs the mean of the local means of a list of distributions, given by: 
    $\overline{m} = \frac1{N} \sum_{i=1}^N m_i$ # type: ignore # type: ignore # type: ignore

    Parameters
    ----------
    dists : list[MNDistribution]
        List of distributions.

    Returns
    -------
    m: np.ndarray
        n-dimensional mean vector.
    """

    means = np.stack([dist.mean for dist in dists])
    return np.mean(means, axis=0)


def calc_C_overline(dists: list[MNDistribution]) -> np.ndarray:
    """Calcs the mean of the local covariance of a list of distributions, given by:
    $\overline{C} = \frac1{N} \sum_{i=1}^N C_i$

    Parameters
    ----------
    dists : list[MNDistribution]
        List of distributions.

    Returns
    -------
    C: np.ndarray
        `n*n`-dimensional covariance matrix.
    """

    covs = np.stack([dist.cov for dist in dists])
    return np.mean(covs, axis=0)


def calc_C_m(dists: list[MNDistribution], m_overline: np.ndarray) -> np.ndarray:
    """Calcs the covariance of the local means of a list of distributions, given by:
    $C_m = \frac1{N} \sum_{i=1}^N (m_i - \overline{m})(m_i - \overline{m})^T = 
    \frac1{N} \sum_{i=1}^N m_i m_i^T - \overline{m} \overline{m}^T$

    Parameters
    ----------
    dists : list[MNDistribution]
        List of distributions.
    
    m_overline : np.ndarray
        Mean of local means.

    Returns
    -------
    C: np.ndarray
        `n*n`-dimensional covariance matrix.
    """

    cov_means = np.stack([np.outer(dist.mean, dist.mean) for dist in dists])
    return np.mean(cov_means, axis=0) - np.outer(m_overline, m_overline)


def calc_uncertain_global_mean(dists: list[MNDistribution]) -> MNDistribution:
    """Calcs the uncertain global mean for the special case, that the input PDF are normal
    distributions. The uncertain global mean is a normal distribution.

    Parameters
    ----------
    dists : list[MNDistribution]
        List of distributions.

    Returns
    -------
    dist : MNDistribution
        Uncertain global mean given as a n-dimensional normal distribution.
    """
    
    N = len(dists)
    m = calc_m_overline(dists)
    M = 1/N * calc_C_overline(dists)
    return MNDistribution(m, M, "uncertain global mean")


def calc_uncertain_global_covariance(dists: list[MNDistribution]) -> MNDistribution:
    """Calcs the uncertain global covariance for the special case, that the input PDF are normal
    distributions. The uncertain global covariance is a normal distribution.

    Parameters
    ----------
    dists : list[MNDistribution]
        List of distributions.

    Returns
    -------
    dist : MNDistribution
        Uncertain global mean given as a r-dimensional normal distribution.
    """

    # Calc mean of uncertain global covariance:
    N = len(dists)
    n = dists[0].dim()
    r = (n*(n+1)) // 2

    m_overline = calc_m_overline(dists)
    C_overline = calc_C_overline(dists)
    C_m = calc_C_m(dists, m_overline)
    C = C_m + (N - 1)/N * C_overline

    T, d = calc_mandel_T_d(n)
    m_C = to_mandel_notation(C, T, d)

    # Calc covariance of uncertain global covariance:
    # Calc entries of CC:
    C_C = np.zeros((r, r))
    for j in range(C_C.shape[0]):
        for k in range(C_C.shape[1]):

            # Calc (i independent) s_2:
            s_2 = sum([
                C_overline[T[j,0], T[k,0]] * C_overline[T[j,1], T[k,1]],
                C_overline[T[j,0], T[k,1]] * C_overline[T[j,1], T[k,0]],
                C_overline[T[j,1], T[k,1]] * C_overline[T[j,0], T[k,0]],
                C_overline[T[j,1], T[k,0]] * C_overline[T[j,0], T[k,1]],
            ])
            s_1 = 0.5 * s_2

            # Calc (i dependent) s_3 & s_4:
            for i in range(N):
                C_i = dists[i].cov
                m_ii = dists[i].mean - m_overline
                s_3 = sum([
                    C_i[T[j,0], T[k,0]] * m_ii[T[j,1]] * m_ii[T[k,1]],
                    C_i[T[j,0], T[k,1]] * m_ii[T[j,1]] * m_ii[T[k,0]],
                    C_i[T[j,1], T[k,1]] * m_ii[T[j,0]] * m_ii[T[k,0]],
                    C_i[T[j,1], T[k,0]] * m_ii[T[j,0]] * m_ii[T[k,1]],
                ])
                s_4 = sum([
                    C_i[T[j,0], T[k,0]] * C_i[T[j,1], T[k,1]],
                    C_i[T[j,0], T[k,1]] * C_i[T[j,1], T[k,0]],
                    C_i[T[j,1], T[k,1]] * C_i[T[j,0], T[k,0]],
                    C_i[T[j,1], T[k,0]] * C_i[T[j,0], T[k,1]],
                ])
                s_1 += s_3 + (N-2)/(2*N) * s_4
            
            # Calc entry of CC:
            C_C[j,k] = (d[j]*d[k])/(N*N) * s_1
    
    return MNDistribution(m_C, C_C)



""" ==================================================
        Projection utilities
    ================================================== """

def align_direction_eigenvecs(eigenvecs: list[np.ndarray]) -> list[np.ndarray]:
    """Aligns a given list of 2x2 matrices, containing eigenvectors, by multiplying them with -1 
    such that all vectors lie within the same half-plane.

    Parameters
    ----------
    eigenvecs : list[np.ndarray]
        List of 2x2 matrices containing eigenvectors (column-wise).

    Returns
    -------
    eigenvecs : list[np.ndarray]
        List of aligned copies of the given 2x2 matrices.
    """
    
    # Note: Column eigenvector[:,0] is first and eigenvector[:,1] is second eigenvector
    # Use first eigenvecs as reference vectors:
    dir_vecs = eigenvecs[0].copy()

    # Iterate over all eigenvecs and flip if necessary:
    for ev in eigenvecs[1:]:
        for i in range(ev.shape[1]):
            
            # check if eigenvector should be flipped:
            d_angle = np.acos(np.dot(dir_vecs[:,i], ev[:,i]) / (np.linalg.norm(dir_vecs[:,i] * np.linalg.norm(ev[:,i]))))
            if d_angle > np.pi / 2:
                ev[:,i] *= -1
            
            # add result to reference vector:
            dir_vecs[:,i] += ev[:,i]
    
    return eigenvecs


def projection_onto_subspace(subspace: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Calcs the projection of a given matrix of points onto a subspace, given by a matrix.

    Parameters
    ----------
    subspace : np.ndarray
        Matrix, spanning the subspace.
    
    data : np.ndarray
        Points to be projected.

    Returns
    -------
    data : np.ndarray
        Projected points.
    """

    if subspace.ndim == 1:
        subspace_norm = (subspace / np.linalg.norm(subspace)).reshape((subspace.shape[0], 1))
    else:
        subspace_norm = subspace / np.linalg.norm(subspace, axis=0)
    return data @ subspace_norm

