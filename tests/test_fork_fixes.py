"""Regression tests for the fitzyracing-bert-score fork (360 Bench).

Tokenizers that don't declare a maximum length (DeBERTa-v3 and others) get transformers'
sentinel ``model_max_length`` of int(1e30). bert-score 0.3.13 passed it to the Rust
("fast") tokenizer as ``max_length``, which raises
``OverflowError: int too big to convert``; on transformers 5 every tokenizer is a fast one.

These tests use the tiny ``hf-internal-testing/tiny-random-DebertaV2Model`` (a few MB) and
remove its ``model_max_length`` to reproduce the sentinel, so no large download is needed.
Set BERT_SCORE_TEST_REAL_MODEL=1 to also run against microsoft/deberta-v3-xsmall.
"""
import json
import os
import shutil

import pytest
import torch
from transformers import AutoTokenizer

import bert_score
from bert_score.utils import sent_encode

TINY = "hf-internal-testing/tiny-random-DebertaV2Model"
SENTINEL = int(1e30)  # transformers' VERY_LARGE_INTEGER
LONG = " ".join(["word"] * 700)  # more than 512 tokens

CANDS = ["hello world", "the cat sat on the mat", "28-year-old chef found dead in San Francisco mall"]
REFS = ["hello world", "a dog ran in the park", "28-Year-Old Chef Found Dead at San Francisco Mall"]


@pytest.fixture(scope="module")
def tiny_without_max_length(tmp_path_factory):
    """A local copy of the tiny model whose tokenizer declares no model_max_length."""
    from huggingface_hub import snapshot_download

    src = snapshot_download(TINY, allow_patterns=["*.json", "*.model", "pytorch_model.bin"])
    dst = tmp_path_factory.mktemp("tiny-deberta-no-max-len")
    for name in os.listdir(src):
        shutil.copy(os.path.join(src, name), dst / name)
    cfg_path = dst / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("model_max_length", None)
    cfg_path.write_text(json.dumps(cfg))
    return str(dst)


@pytest.fixture
def fast_tokenizer():
    tok = AutoTokenizer.from_pretrained(TINY, use_fast=True)
    assert tok.is_fast
    return tok


def test_sent_encode_sentinel_max_length_does_not_overflow(fast_tokenizer):
    fast_tokenizer.model_max_length = SENTINEL
    ids = sent_encode(fast_tokenizer, "hello world")
    assert ids == fast_tokenizer.encode("hello world", add_special_tokens=True)


def test_sent_encode_sentinel_does_not_truncate(fast_tokenizer):
    # Same as the slow (Python) tokenizer did with the sentinel: no limit, no truncation.
    fast_tokenizer.model_max_length = SENTINEL
    ids = sent_encode(fast_tokenizer, LONG)
    assert len(ids) == 702
    assert ids == fast_tokenizer.encode(LONG, add_special_tokens=True)
    # The tokenizer itself is not modified.
    assert fast_tokenizer.model_max_length == SENTINEL


def test_sent_encode_real_max_length_still_truncates(fast_tokenizer):
    # Unchanged behaviour: a declared model_max_length still truncates.
    assert fast_tokenizer.model_max_length == 512
    assert len(sent_encode(fast_tokenizer, LONG)) == 512
    fast_tokenizer.model_max_length = 16
    assert len(sent_encode(fast_tokenizer, LONG)) == 16


@pytest.mark.parametrize("model", ["microsoft/deberta-xlarge-mnli", "microsoft/deberta-v3-base"])
def test_recommended_deberta_tokenizers(model):
    # Only the tokenizer files are downloaded (about 1-2 MB). deberta-xlarge-mnli is the model
    # the bert-score README recommends; neither tokenizer declares a model_max_length.
    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    assert tok.model_max_length == SENTINEL
    assert sent_encode(tok, "hello world") == tok.encode("hello world", add_special_tokens=True)


def test_score_with_tokenizer_without_max_length_fast(tiny_without_max_length):
    tok = AutoTokenizer.from_pretrained(tiny_without_max_length, use_fast=True)
    assert tok.model_max_length == SENTINEL
    P, R, F = bert_score.score(CANDS, REFS, model_type=tiny_without_max_length, num_layers=2,
                               use_fast_tokenizer=True, batch_size=2)
    assert F.shape == (3,)
    assert torch.isfinite(F).all()
    assert F[0].item() == pytest.approx(1.0, abs=1e-5)


def test_score_with_tokenizer_without_max_length_default(tiny_without_max_length):
    # use_fast_tokenizer=False is the default; on transformers 5 the tokenizer is fast anyway.
    P, R, F = bert_score.score(CANDS, REFS, model_type=tiny_without_max_length, num_layers=2)
    assert torch.isfinite(F).all()


def test_fast_tokenizer_scores_match_slow_tokenizer(tiny_without_max_length):
    # The slow tokenizer never had the problem; the fixed fast path must give the same scores.
    from transformers import __version__ as tv
    from packaging import version

    if version.parse(tv) >= version.parse("5.0.0"):
        pytest.skip("transformers 5 has no slow tokenizers")
    slow = bert_score.score(CANDS, REFS, model_type=tiny_without_max_length, num_layers=2,
                            use_fast_tokenizer=False)
    fast = bert_score.score(CANDS, REFS, model_type=tiny_without_max_length, num_layers=2,
                            use_fast_tokenizer=True)
    for a, b in zip(slow, fast):
        assert torch.allclose(a, b, atol=1e-6)


def test_scorer_class_with_tokenizer_without_max_length(tiny_without_max_length):
    scorer = bert_score.BERTScorer(model_type=tiny_without_max_length, num_layers=2,
                                   use_fast_tokenizer=True)
    P, R, F = scorer.score(CANDS[:2], REFS[:2])
    assert torch.isfinite(F).all()


@pytest.mark.skipif(not os.environ.get("BERT_SCORE_TEST_REAL_MODEL"),
                    reason="set BERT_SCORE_TEST_REAL_MODEL=1 (downloads microsoft/deberta-v3-xsmall)")
@pytest.mark.parametrize("use_fast", [False, True])
def test_real_deberta_v3(use_fast):
    # DeBERTa-v3 uses relative positions, so a >512-token input is scored in full, as the slow
    # tokenizer always did.
    P, R, F = bert_score.score(CANDS + [LONG], REFS + [LONG + " again"], model_type="microsoft/deberta-v3-xsmall",
                               use_fast_tokenizer=use_fast)
    assert torch.isfinite(F).all()
    # transformers 5 loads this checkpoint in float16 (its stored dtype), so allow 1e-3.
    assert F[0].item() == pytest.approx(1.0, abs=1e-3)
