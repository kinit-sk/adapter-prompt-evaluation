import tasks
from tasks.dataset import Dataset
from tasks.tasks import MLQA, SlovakSQuAD, CSSQuAD, WikiANN


def dataset_factory(
    dataset_name: str, **kwargs
) -> Dataset:
    dataset = {
        "mlqa": MLQA,
        "squad": MLQA,
        "sksquad": SlovakSQuAD,
        "cssquad": CSSQuAD,
        'wikiann': WikiANN
    }[dataset_name]
    return dataset(**kwargs)
