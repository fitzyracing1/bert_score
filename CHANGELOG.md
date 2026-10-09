# Changelog

## 0.3.14 (2026-10-09, fitzyracing-bert-score fork)

Based on bert-score 0.3.13. Distribution name `fitzyracing-bert-score`; the import names
(`bert_score`, `bert_score_cli`) and the `bert-score` / `bert-score-show` commands are unchanged.

- Fix `OverflowError: int too big to convert` for models whose tokenizer declares no
  `model_max_length` (DeBERTa-v3 models such as `microsoft/deberta-v3-base`, mDeBERTa, ...)
  with fast tokenizers, which on transformers 5 means always (upstream #205, PR #206, and
  huggingface/evaluate#739). Such tokenizers now encode without truncation, exactly as the slow
  tokenizers did, so scores match bert-score 0.3.13 with the slow tokenizer on transformers 4.
  Tokenizers that declare a maximum length truncate as before.
- `bert_score.__version__` still reads `"0.3.12"`, as in upstream 0.3.13 (which never bumped it),
  so the hash strings returned with `return_hash=True` are unchanged.
- Packaging: `pyproject.toml` replaces `setup.py`.
- Tests: add `tests/test_fork_fixes.py` (uses a tiny public test model, a few MB).

## 0.3.13 (2023-02-20) and earlier

See the "News" section of the README and the
[upstream history](https://github.com/Tiiiger/bert_score/commits/master).
