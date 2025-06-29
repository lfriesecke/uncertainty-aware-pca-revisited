import numpy as np

from pathlib import Path

from ..distribution import MNDistribution
from ..utils.pca import calc_C



def _dist_from_data(data: np.ndarray, **kwargs) -> MNDistribution:
    """ Creates distribution from given array (and optional 'name'). """
    mean = np.mean(data, axis=0)
    cov = calc_C(data)
    name = kwargs.get("name")
    return MNDistribution(mean, cov) if name == None else MNDistribution(mean, cov, name)


def dataset_iris() -> list[MNDistribution]:
    """Reads iris dataset file and creates corresponding distributions.

    Returns:
        list[MNDistribution]: List of distributions.
    """
    
    # Read data from file:
    with open(Path(__file__).with_name("iris.csv")) as f:
        lines = f.readlines()
    
    # Process data:
    iris_list = []
    sublist = []
    iris_type = 'Iris-setosa\n'
    for l in lines:
        vals = l.split(',')

        # Check if new distribution has been reached:
        if iris_type != vals[-1]:
            iris_type = vals[-1]
            iris_list += [(iris_type[:-1], sublist)]
            sublist = []

        # Add data to list:
        sublist += [[float(v) for v in vals[:-1]]]
    iris_list += [(iris_type[:-1], sublist)]

    # Generate distributions:
    dists = []
    for name, data in iris_list:
        data_np = np.array(data)
        dists += [_dist_from_data(data_np, **{"name" : name})]

    # Return distributions:
    return dists


def dataset_students_grades() -> list[MNDistribution]:
    """Reads student grades dataset file and creates corresponding distributions.

    Returns:
        list[MNDistribution]: List of distributions.
    """

    # Read data from file:
    with open(Path(__file__).with_name("students_grades.csv")) as f:
        lines = f.readlines()

    # Create distributions:
    dists = []
    for i, line in enumerate(lines):
        vals = line.split(',')
        label = vals[0]
        mean = np.array([float(v) for v in vals[1:5]])
        cov = np.diag([float(v) for v in vals[5:]])
        dists.append(MNDistribution(mean, cov, label))
    
    # Return distributions:
    return dists


def dataset_anuran_calls() -> list[MNDistribution]:
    """Reads anuran calls dataset file and creates corresponding distributions.

    Returns:
        list[MNDistribution]: List of distributions.
    """

    # Read data from file:
    with open(Path(__file__).with_name("anuran_calls.csv")) as f:
        lines = f.readlines()
    
    # Process data:
    anuran_list = []
    sublist = []
    anuran_type = 'Leptodactylidae'
    for l in lines[1:]:
        vals = l.split(',')

        # Check if new distribution has been reached:
        if anuran_type != vals[-4]:
            anuran_list += [(anuran_type, sublist)]
            anuran_type = vals[-4]
            sublist = []

        # Add data to list:
        sublist += [[float(v) for v in vals[:-4]]]
    anuran_list += [(anuran_type, sublist)]

    # Aggregate lists:
    names = []
    [names.append(name) for (name, _) in anuran_list if name not in names]
    anuran_dict = {name : [] for name in names}
    for name, data in anuran_list:
        anuran_dict[name] += data

    # Generate distributions:
    dists = []
    for name in names:
        data_np = np.array(anuran_dict[name])
        dists += [_dist_from_data(data_np, **{"name" : name})]

    # Return distributions:
    return dists
