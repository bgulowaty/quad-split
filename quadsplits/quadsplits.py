import numpy as np
from loguru import logger as log
from box import Box
import problexity as px

def should_pass(cutoff_samples, min_split_size):
    left_size = len(cutoff_samples.left.x)
    right_size = len(cutoff_samples.right.x)

    return left_size >= min_split_size and right_size >= min_split_size


def find_best_cutoff_for(dataset, dimension, minimal_split_percentage=0.1, cutoff_function=px.f2):
    samples_in_cutoffs = get_samples_in_cutoff(dataset, dimension)

    min_split_size = len(dataset.x) * minimal_split_percentage
    log.info("Min split size = {}", min_split_size)
    log.info("Min split percentage = {}", minimal_split_percentage)

    samples_in_cutoffs_filtered = filter_by_min_split_size(min_split_size, samples_in_cutoffs)

    log.trace("Cutoffs filtered = {}", len(samples_in_cutoffs_filtered))

    for it in samples_in_cutoffs_filtered:
        try:
            log.trace("Computing complexity for {}", it)
            left_complexities = compute_complexities_ovo(cutoff_function, it.left)
            right_complexities = compute_complexities_ovo(cutoff_function, it.right)

        except Exception as e:
            log.exception(e)
            return None

        it.left_complexity = np.mean(left_complexities)
        it.right_complexity = np.mean(right_complexities)

    if not samples_in_cutoffs_filtered:
        return None

    lowest_complexity = min(samples_in_cutoffs_filtered, key=lambda it: it.left_complexity + it.right_complexity)

    if lowest_complexity is None:
        return None

    return lowest_complexity.cutoff, lowest_complexity.left_complexity + lowest_complexity.right_complexity


def compute_complexities_ovo(cutoff_function, dataset):
    complexities = []
    for label in np.unique(dataset.y):
        log.debug("Main label {}", label)
        ovo_y = dataset.y.copy()
        ovo_y[ovo_y != label] = 0
        ovo_y[ovo_y == label] = 1
        if len(np.unique(ovo_y)) != 1:
            complexity = cutoff_function(dataset.x, ovo_y)
            complexities.append(complexity)
        else:
            complexities.append(0)
    return complexities

def recursive_cutoff(dataset, current_conditions=None, recursion_level=0, min_samples = 10, recursion_limit = 4, minimal_split_percentage = 0.1, complexity_measure = px.f2):
    log.info("Recursion level = {}", recursion_level)

    if current_conditions is None:
        current_conditions = list()

    if recursion_limit != -1 and recursion_level >= recursion_limit:
        log.info("Recursion limit reached {}", recursion_level)
        return {" and ".join(current_conditions)}

    recursion_level = recursion_level + 1
    log.debug("Recursion level = {}", recursion_level)

    if len(dataset.x) < min_samples:
        log.info("min_samples limit reached {} < {}", len(dataset.x), min_samples)
        return {" and ".join(current_conditions)}

    features_count = dataset.x.shape[1]

    best_cutoff_by_dimension = {}

    for feature_idx in range(features_count):
        cutoff_and_value = find_best_cutoff_for(Box(x=dataset.x, y=dataset.y), feature_idx, minimal_split_percentage=minimal_split_percentage, cutoff_function=complexity_measure)
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

    left_statements = recursive_cutoff(Box(x=dataset.x[left_indicies], y=dataset.y[left_indicies]), current_conditions + [left_conditions], recursion_level + 1)
    right_statements = recursive_cutoff(Box(x=dataset.x[right_indicies], y=dataset.y[right_indicies]), current_conditions + [right_conditions], recursion_level + 1)
    log.debug("Left statements {}", left_statements)
    log.debug("Right statments {}", right_statements)

    return left_statements.union(right_statements)


def filter_by_min_split_size(min_split_size, samples_in_cutoffs):
    samples_in_cutoffs_filtered = [
        cutoff_samples for cutoff_samples in samples_in_cutoffs if should_pass(cutoff_samples, min_split_size)
    ]
    return samples_in_cutoffs_filtered


def get_samples_in_cutoff(dataset, dimension):
    possible_cutoffs = set(sorted(dataset.x[:, dimension]))

    log.info("Feature = {}", dimension)
    log.trace("Possible cutoffs = {}", len(possible_cutoffs))
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
