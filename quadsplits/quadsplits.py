import numpy as np
from loguru import logger as log
from box import Box
import problexity as px

NO_COMPLEXITY = 0

def both_sides_have_more_or_equal_samples(cutoff_samples, min_split_size):
    left_size = len(cutoff_samples.left.x)
    right_size = len(cutoff_samples.right.x)

    return left_size >= min_split_size and right_size >= min_split_size


def find_best_cutoff_for(dataset, dimension, minimal_split_percentage=0.1, cutoff_function=px.f2, oversampling_function= lambda x,y: (x,y)):
    samples_in_cutoffs = get_samples_in_cutoff(dataset, dimension)

    min_split_size = len(dataset.x) * minimal_split_percentage
    log.debug("min split percentage={}, size={}", minimal_split_percentage, min_split_size)

    samples_filtered_by_min_split_size = filter_by_min_split_size(min_split_size, samples_in_cutoffs)
    log.debug("len(cutoffs_filtered)={}", len(samples_filtered_by_min_split_size))

    if not samples_filtered_by_min_split_size:
        return None

    for it in samples_filtered_by_min_split_size:
        # log.trace("Computing complexity for {}", it)

        x_oversampled, y_oversampled = oversampling_function(it.left.x, it.left.y)
        try:
            left_complexities = cutoff_function(x_oversampled, y_oversampled)
            right_complexities = cutoff_function(x_oversampled, y_oversampled)

        except Exception as e:
            log.exception(e)
            return None

        it.left_complexity = left_complexities
        it.right_complexity = right_complexities

    lowest_complexity_by_sum = min(samples_filtered_by_min_split_size,
                                   key=lambda it: it.left_complexity + it.right_complexity)

    if lowest_complexity_by_sum is None:
        return None

    return lowest_complexity_by_sum.cutoff, lowest_complexity_by_sum.left_complexity + lowest_complexity_by_sum.right_complexity



def recursive_cutoff(dataset, current_conditions=None, recursion_level=0, min_samples=10, recursion_limit=4,
                     minimal_split_percentage=0.1, complexity_measure=px.f2, oversampling_function=lambda x,y: (x,y)):
    log.debug("Recursion level = {}", recursion_level)

    if current_conditions is None:
        current_conditions = list()

    if recursion_limit != -1 and recursion_level >= recursion_limit:
        log.debug("recursion limit reached={}", recursion_level)
        return {" and ".join(current_conditions)}

    recursion_level = recursion_level + 1
    log.debug("Recursion level = {}", recursion_level)

    if len(dataset.x) < min_samples:
        log.debug("min_samples limit reached {} < {}", len(dataset.x), min_samples)
        return {" and ".join(current_conditions)}

    features_count = dataset.x.shape[1]
    best_cutoff_by_dimension = {}

    for feature_idx in range(features_count):
        cutoff_and_value = find_best_cutoff_for(Box(x=dataset.x, y=dataset.y), feature_idx,
                                                minimal_split_percentage=minimal_split_percentage,
                                                cutoff_function=complexity_measure,
                                                oversampling_function=oversampling_function)
        if cutoff_and_value is None:
            continue

        cutoff, value = cutoff_and_value

        best_cutoff_by_dimension[feature_idx] = {
            'cutoff': cutoff,
            'value': value
        }

    log.debug("Cutoffs = {}", best_cutoff_by_dimension)
    if not best_cutoff_by_dimension:
        log.debug("No best cutoff found")
        return {" and ".join(current_conditions)}

    best_cutoff_entry = min(best_cutoff_by_dimension.items(), key=lambda it: it[1]['cutoff'])
    best_cutoff_dimension = best_cutoff_entry[0]
    best_cutoff = best_cutoff_entry[1]['cutoff']
    best_cutoff_value = best_cutoff_entry[1]['value']

    log.debug("Best cutoff value = {} ({} at dim {})", best_cutoff_value, best_cutoff, best_cutoff_dimension)

    left_conditions = f"col{best_cutoff_dimension} <= {best_cutoff}"
    right_conditions = f"col{best_cutoff_dimension} > {best_cutoff}"
    log.debug(left_conditions)
    log.debug(right_conditions)

    left_indicies = dataset.x[:, best_cutoff_dimension] <= best_cutoff
    right_indicies = dataset.x[:, best_cutoff_dimension] > best_cutoff

    left_statements = recursive_cutoff(Box(x=dataset.x[left_indicies], y=dataset.y[left_indicies]),
                                       current_conditions=current_conditions + [left_conditions],
                                       recursion_level=recursion_level + 1,
                                       min_samples=min_samples,
                                       recursion_limit=recursion_limit,
                                       minimal_split_percentage=minimal_split_percentage,
                                       complexity_measure=complexity_measure,
                                       oversampling_function=oversampling_function)
    right_statements = recursive_cutoff(Box(x=dataset.x[right_indicies], y=dataset.y[right_indicies]),
                                        current_conditions=current_conditions + [right_conditions],
                                        recursion_level=recursion_level + 1,
                                        min_samples=min_samples,
                                        recursion_limit=recursion_limit,
                                        minimal_split_percentage=minimal_split_percentage,
                                        complexity_measure=complexity_measure,
                                        oversampling_function=oversampling_function)
    log.debug("Left statements {}", left_statements)
    log.debug("Right statments {}", right_statements)

    return left_statements.union(right_statements)


def filter_by_min_split_size(min_split_size, samples_in_cutoffs):
    samples_in_cutoffs_filtered = [
        cutoff_samples for cutoff_samples in samples_in_cutoffs if both_sides_have_more_or_equal_samples(cutoff_samples, min_split_size)
    ]
    return samples_in_cutoffs_filtered


def get_samples_in_cutoff(dataset, dimension):
    possible_cutoffs = set(sorted(dataset.x[:, dimension]))

    log.trace("Feature = {}, len(possible_cutoffs) = {}", dimension, len(possible_cutoffs))
    samples_in_cutoffs = [
        Box({
            'cutoff': cutoff,
            'left': {
                'x': dataset.x[dataset.x[:, dimension] <= cutoff],
                'y': dataset.y[dataset.x[:, dimension] <= cutoff],
            },
            'right': {
                'x': dataset.x[dataset.x[:, dimension] > cutoff],
                'y': dataset.y[dataset.x[:, dimension] > cutoff],
            }
        })
        for cutoff in possible_cutoffs
    ]

    return [it for it in samples_in_cutoffs
            if len(it.left.x) != 0 and len(it.right.x) != 0]
