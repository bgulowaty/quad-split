import pytest
from box import Box
from .quadsplits import get_samples_in_cutoff, both_sides_have_more_or_equal_samples, compute_complexities_ovo, find_best_cutoff_for
import numpy as np
from numpy.random import random_sample

X_SUMMING_CUTOFF_FUNCTION = lambda x, y: np.sum(x[y == 1, :])


@pytest.mark.parametrize("dataset,dimension,expected_splits", [
    [{"x": np.array([[1, 1], [1, 2], [1, 15], [1, 22], [1, 25]]),
      "y": np.array([0, 0, 0, 1, 1])
      }, 1,
     [
         [[[1, 1]], [[1, 2], [1, 15], [1, 22], [1, 25]]],
         [[[1, 1], [1, 2]], [[1, 15], [1, 22], [1, 25]]],
         [[[1, 1], [1, 2], [1, 15]], [[1, 22], [1, 25]]],
         [[[1, 1], [1, 2], [1, 15], [1, 22]], [[1, 25]]],
     ]],
    [{"x": np.array([[1, 1], [1, 2], [1, 15], [1, 22], [1, 25]]),
      "y": np.array([0, 0, 0, 1, 1])
      }, 0,
     []],
    [{"x": np.array([[1, 1], [1, 2], [2, 15], [2, 22], [2, 25]]),
      "y": np.array([0, 0, 0, 1, 1])
      }, 0,
     [
         [[[1, 1], [1, 2]], [[2, 15], [2, 22], [2, 25]]],
     ]],
])
def test_get_samples_in_cutoff(dataset, dimension, expected_splits):
    # given
    dataset = Box(dataset)

    # when
    samples_in_cutoffs = get_samples_in_cutoff(dataset, dimension)

    # then
    for idx, samples_in_cutoff in enumerate(samples_in_cutoffs):
        print(idx)
        print(samples_in_cutoff)
        assert (samples_in_cutoff.left.x == expected_splits[idx][0]).all()
        assert (samples_in_cutoff.right.x == expected_splits[idx][1]).all()


@pytest.mark.parametrize("cutoff_samples,min_split_size,expected", [
    [{"left": {"x": random_sample(size=(10, 2)), "y": random_sample(size=(10, 2))},
      "right": {"x": random_sample(size=(10, 2)), "y": random_sample(size=(10, 2))}},
     1.25, True],
    [{"left": {"x": random_sample(size=(10, 2)), "y": random_sample(size=(10, 2))},
      "right": {"x": random_sample(size=(10, 2)), "y": random_sample(size=(10, 2))}},
     10, True],
    [{"left": {"x": random_sample(size=(10, 2)), "y": random_sample(size=(10, 2))},
      "right": {"x": random_sample(size=(10, 2)), "y": random_sample(size=(10, 2))}},
     10.25, False],
    [{"left": {"x": np.array([]), "y": np.array([])},
      "right": {"x": random_sample(size=(10, 2)), "y": random_sample(size=(10, 2))}},
     2, False]
])
def test_should_pass(cutoff_samples, min_split_size, expected):
    # given
    cutoff_samples = Box(cutoff_samples)

    # when
    actual = both_sides_have_more_or_equal_samples(cutoff_samples, min_split_size)

    # then
    assert actual == expected



# def test_find_best_cutoff_for():
#     # given
#     dataset = Box(
#         x=np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
#         y=np.array([1, 2, 2, 1])
#     )
#     dimension = 0
#     minimal_split_percentage = 0.1
#
#     # when
#     actual_cutoff, actual_complexities_sum = find_best_cutoff_for(dataset, dimension, minimal_split_percentage,
#                                                                   X_SUMMING_CUTOFF_FUNCTION)
#
#     # then
#     assert actual_cutoff == 3
#     assert actual_complexities_sum == 10 # left: (2 + 4)/2 right: (6 + 8)/2
