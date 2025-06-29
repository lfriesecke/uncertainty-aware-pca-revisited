import numpy as np



def calc_mandel_T_d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcs the auxiliary matrix T and the vector d for converting a n*n matrix into a n*(n+1)/2-dimensional
    vector in mandel notation.
    """

    # Calc r:
    r = n * (n + 1) // 2

    # Create matrix T:
    T = np.zeros((r, 2), np.int32)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            T[idx,0], T[idx,1] = i, j
            idx += 1
    
    # Create vector d:
    d = np.array([1 if T[i,0] == T[i,1] else np.sqrt(2) for i in range(r)])
    return (T, d)


def to_mandel_notation(matrix: np.ndarray, T: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Converts a n*n matrix into a n*(n+1)/2-dimensinonal vector in mandel notation, given by
    $v(C)[i] = d[i] \cdot C[T[i,1], T[i,2]]$
    """
    r, _ = T.shape
    vC = np.array([d[i] * matrix[T[i,0], T[i,1]] for i in range(r)])
    return vC
