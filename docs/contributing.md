# Contributing

This fork only takes fixes for confirmed upstream bugs, and each fix needs a test that fails on the original (see the 360 Bench rules). Behaviour should stay the same as upstream wherever upstream did not crash. Before you open a PR, run:

```
pytest tests/test_fork_fixes.py                               # tiny test model
BERT_SCORE_TEST_REAL_MODEL=1 pytest tests/test_fork_fixes.py  # also deberta-v3-xsmall
```

If upstream ships the fix itself, the fork will point people back to upstream.
