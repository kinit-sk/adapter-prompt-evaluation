import tasks
from tasks.dataset import Dataset
from tasks.tasks import MLQA, SlovakSQuAD, CSSQuAD


def dataset_factory(
    dataset_name: str,
) -> Dataset:
    dataset = {
        "mlqa": MLQA,
        "squad": MLQA,
        "sksquad": SlovakSQuAD,
        "cssquad": CSSQuAD,
    }[dataset_name]
    return dataset()
