# Soft Language Prompts for Language Transfer

This is the source code for the paper [_Soft Language Prompts for Language Transfer_](https://aclanthology.org/2025.naacl-long.517/).

## Abstract

Cross-lingual knowledge transfer, especially between high- and low-resource languages, remains challenging in natural language processing (NLP). This study offers insights for improving cross-lingual NLP applications through the combination of parameter-efficient fine-tuning methods. We systematically explore strategies for enhancing cross-lingual transfer through the incorporation of language-specific and task-specific adapters and soft prompts. We present a detailed investigation of various combinations of these methods, exploring their efficiency across 16 languages, focusing on 10 mid- and low-resource languages. We further present to our knowledge the first use of soft prompts for language transfer, a technique we call **soft language prompts**. Our findings demonstrate that in contrast to claims of previous work, a combination of language and task adapters does not always work best; instead, combining a soft language prompt with a task adapter outperforms most configurations in many cases.

## Reproducibility

### Installation

To replicate our experiments, ensure you have Python version 3.11.8 and install the required packages listed in `requirements.txt` using the following command:

```bash
pip install -r requirements.txt
```

### Reproduction of the results

We have prepared several scripts to facilitate the training of language and task-specific adapters and soft prompts. The following sections describe how to use these scripts.

Ensure that all scripts are executed from the `src` folder.

#### Language representations

To train language representations, run:

```bash
python -m scripts.lm
```

This script (scripts/lm.py) trains the language-specific adapters and soft prompts using the Wikipedia dump available on [HuggingFace](https://huggingface.co/datasets/wikimedia/wikipedia). The exact version of the dump used is specified within the script.

#### Task representations

We have multiple scripts to train task-specific adapters and soft prompts, as we evaluated various configurations of these methods.

Available scripts include:

- `scripts/qa` - for training task-specific adapters and soft prompts for the question answering task
- `scripts/ner` - for training task-specific adapters and soft prompts for the named-entity recognition task
- `scripts/xnli` - for training task-specific adapters and soft prompts for the natural language inference task
- `scripts/checkworthy` - for training task-specific adapters and soft prompts for check-worthy claim detection

For example, to run the script for question answering, use:

```bash
python -m scripts.qa
```

#### Cross-lingual evaluation

In our experiments, we assess cross-lingual performance by training task representations in one language and evaluating them in other languages. For this evaluation, use the script  `scripts/crosslingual.py`.

To evaluate trained task representations on English data and test them on other languages, run:

```bash
python -m scripts.crosslingual --trained_language english --test_languages german spanish slovak czech telugu
```

## Paper citing

If you use the code or information from this repository, please cite our paper, which is available in [ACL Anthology](https://aclanthology.org/2025.naacl-long.517/).

```bibtex
@inproceedings{vykopal-etal-2025-soft,
    title = "Soft Language Prompts for Language Transfer",
    author = "Vykopal, Ivan  and
      Ostermann, Simon  and
      Simko, Marian",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.517/",
    pages = "10294--10313",
    ISBN = "979-8-89176-189-6",
    abstract = "Cross-lingual knowledge transfer, especially between high- and low-resource languages, remains challenging in natural language processing (NLP). This study offers insights for improving cross-lingual NLP applications through the combination of parameter-efficient fine-tuning methods. We systematically explore strategies for enhancing cross-lingual transfer through the incorporation of language-specific and task-specific adapters and soft prompts. We present a detailed investigation of various combinations of these methods, exploring their efficiency across 16 languages, focusing on 10 mid- and low-resource languages. We further present to our knowledge the first use of soft prompts for language transfer, a technique we call soft language prompts. Our findings demonstrate that in contrast to claims of previous work, a combination of language and task adapters does not always work best; instead, combining a soft language prompt with a task adapter outperforms most configurations in many cases."
}
```
