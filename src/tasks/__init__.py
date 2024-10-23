import tasks
from tasks.dataset import Dataset
from tasks.tasks import MLQA, SlovakSQuAD, CSSQuAD, TeQuAD, WikiANN, MultiClaimCheckWorthy, XNLI, ArabicSQuAD, ChineseSQuAD, SberSQuAD, KenSwQuAD, UQA, XQuAD, SlovenianSQuAD, IndicQA


def dataset_factory(
    dataset_name: str, **kwargs
) -> Dataset:
    dataset = {
        "mlqa": MLQA,
        "squad": MLQA,
        "sksquad": SlovakSQuAD,
        "cssquad": CSSQuAD,
        "tequad": TeQuAD,
        "zhsquad": ChineseSQuAD,
        "rusquad": SberSQuAD,
        "swsquad": KenSwQuAD,
        "uqa": UQA,
        'wikiann': WikiANN,
        'xnli': XNLI,
        'multiclaim_checkworthy': MultiClaimCheckWorthy,
        'arabic_squad': ArabicSQuAD,
        'xquad': XQuAD,
        'slosquad': SlovenianSQuAD,
        'indicqa': IndicQA,
    }[dataset_name]
    return dataset(**kwargs)
