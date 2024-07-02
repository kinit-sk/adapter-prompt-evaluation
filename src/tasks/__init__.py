import tasks
from tasks.dataset import Dataset
from tasks.tasks import MLQA, SlovakSQuAD, CSSQuAD, TeQuAD, WikiANN, MultiClaimCheckWorthy, XNLI


def dataset_factory(
    dataset_name: str, **kwargs
) -> Dataset:
    dataset = {
        "mlqa": MLQA,
        "squad": MLQA,
        "sksquad": SlovakSQuAD,
        "cssquad": CSSQuAD,
        "tequad": TeQuAD,
        'wikiann': WikiANN,
        'xnli': XNLI,
        'multiclaim_checkworthy': MultiClaimCheckWorthy
    }[dataset_name]
    return dataset(**kwargs)
