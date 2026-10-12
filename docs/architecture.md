# Architecture

| Path | Role |
|---|---|
| `bert_score/score.py` | `score()` and plotting |
| `bert_score/scorer.py` | `BERTScorer` class |
| `bert_score/utils.py` | tokenization, model loading (the fix lives here) |
| `bert_score/rescale_baseline/` | baseline files |
| `bert_score_cli/` | CLI entry points |
| `tests/` | unittest/pytest suites |


