# create methods for sampling mixture of datasets
from datasets import concatenate_datasets


# examples-proportional mixing
def proportional_mixing(datasets, required_size=524288):
    """
    Mix datasets in proportion to their size.

    datasets: list of datasets to mix (all of them are allready preprocessed)
    """
    # shuffle datasets
    for dataset in datasets:
        dataset.shuffle()

    # get dataset sizes
    dataset_sizes = [len(dataset) for dataset in datasets]
    # check if overall size is sufficient if not return mixture of datasets
    if sum(dataset_sizes) < required_size:
        return concatenate_datasets(datasets)
    # get proportions
    proportions = [size / sum(dataset_sizes) for size in dataset_sizes]
    # get number of samples for each dataset
    num_samples = [int(size * required_size) for size in proportions]
    # get remainder
    remainder = required_size - sum(num_samples)
    # add remainder to first dataset
    num_samples[0] += remainder
    print(num_samples)
    # use dataset concatenation to mix datasets
    mixed_dataset = concatenate_datasets(
        [dataset.select(range(size)) for dataset,
         size in zip(datasets, num_samples)]
    )
    # shuffle mixed dataset
    mixed_dataset = mixed_dataset.shuffle()

    return mixed_dataset


def all_mixing(datasets):
    """
    Mix datasets in proportion to their size.

    datasets: list of datasets to mix (all of them are allready preprocessed)
    """
    # shuffle datasets
    for dataset in datasets:
        dataset.shuffle()

    # use dataset concatenation to mix datasets
    mixed_dataset = concatenate_datasets(datasets)
    return mixed_dataset
