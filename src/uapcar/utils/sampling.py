import numpy as np

from ..distribution import MNDistribution



# Generate list of sample points of set of distributions:
def samples_from_distributions(
        distributions: list[MNDistribution],
        num_samples: int,
        seed: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """Samples a list of multivariate normal distributions.

    Parameters
    ----------
    distributions : list[MNDistribution]
        List of multivariate normal distributions. All distributions have to share the same number of dimensions.

    num_samples : int
        Number of samples to draw from each distribution.
    
    seed : int
        Seed to initialize randomizer. The randomizer then generates a individual seed for each distribution.

    Returns
    -------
    samples : tuple[list[np.ndarray], np.ndarray]
        List with matrix of samples for each distribution and matrix with samples from all distributions.
    """
    
    # Check if all distributions share same number of dimensions:
    d = distributions[0].dim()
    if not all(dist.dim() == d for dist in distributions):
        raise ValueError("All given distributions have to share the same number of dimensions.")
    
    # Generate seeds for each distribution:
    rng = np.random.default_rng(seed=seed)
    seeds = rng.integers(low=0, high=9223372036854775807, size=len(distributions), dtype=np.int64)

    # Generate and return samples:
    sample_list = [dist.samples(num_samples, seeds[i]) for i, dist in enumerate(distributions)]
    sample_matrix = np.concatenate([sample_list], axis=0).reshape(num_samples * len(distributions), distributions[0].dim())
    return sample_list, sample_matrix
