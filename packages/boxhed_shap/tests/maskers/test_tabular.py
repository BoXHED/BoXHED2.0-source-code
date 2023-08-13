""" This file contains tests for the Tabular maskers.
"""

import tempfile
import numpy as np
import boxhed_shap


def test_serialization_independent_masker_dataframe():
    """ Test the serialization of an Independent masker based on a data frame.
    """

    X, _ = boxhed_shap.datasets.boston()

    # initialize independent masker
    original_independent_masker = boxhed_shap.maskers.Independent(X)

    with tempfile.TemporaryFile() as temp_serialization_file:

        # serialize independent masker
        original_independent_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_independent_masker = boxhed_shap.maskers.Independent.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_independent_masker(mask, X[:1].values[0])[1], new_independent_masker(mask, X[:1].values[0])[1])

def test_serialization_independent_masker_numpy():
    """ Test the serialization of an Independent masker based on a numpy array.
    """


    X, _ = boxhed_shap.datasets.boston()
    X = X.values

    # initialize independent masker
    original_independent_masker = boxhed_shap.maskers.Independent(X)

    with tempfile.TemporaryFile() as temp_serialization_file:

        # serialize independent masker
        original_independent_masker.save(temp_serialization_file)


        temp_serialization_file.seek(0)

        # deserialize masker
        new_independent_masker = boxhed_shap.maskers.Masker.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_independent_masker(mask, X[0])[0], new_independent_masker(mask, X[0])[0])

def test_serialization_partion_masker_dataframe():
    """ Test the serialization of a Partition masker based on a DataFrame.
    """

    X, _ = boxhed_shap.datasets.boston()

    # initialize partition masker
    original_partition_masker = boxhed_shap.maskers.Partition(X)

    with tempfile.TemporaryFile() as temp_serialization_file:

        # serialize partition masker
        original_partition_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_partition_masker = boxhed_shap.maskers.Partition.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[:1].values[0])[1], new_partition_masker(mask, X[:1].values[0])[1])

def test_serialization_partion_masker_numpy():
    """ Test the serialization of a Partition masker based on a numpy array.
    """

    X, _ = boxhed_shap.datasets.boston()
    X = X.values

    # initialize partition masker
    original_partition_masker = boxhed_shap.maskers.Partition(X)

    with tempfile.TemporaryFile() as temp_serialization_file:

        # serialize partition masker
        original_partition_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_partition_masker = boxhed_shap.maskers.Masker.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[0])[0], new_partition_masker(mask, X[0])[0])
