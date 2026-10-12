# Getting started

## Install

```
pip uninstall -y bert-score && pip install fitzyracing-bert-score
```

The import name is still `bert_score`. Uninstall the original first, because both packages own the same files.

## Usage

`from bert_score import score` / `BERTScorer`, as upstream. Console scripts: `bert-score` (`bert_score_cli.score:main`) and `bert-score-show` (`bert_score_cli.visualize:main`).

## Develop and test

```
pytest tests/test_fork_fixes.py                               # tiny test model
BERT_SCORE_TEST_REAL_MODEL=1 pytest tests/test_fork_fixes.py  # also deberta-v3-xsmall
```
