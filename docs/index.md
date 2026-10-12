# fitzyracing-bert-score

`fitzyracing-bert-score` is a drop-in fork of [BERTScore (bert-score)](https://github.com/Tiiiger/bert_score) and part of [360 Bench](https://github.com/fitzyracing1/360-bench). It is based on upstream `bert-score==0.3.13`, and the API is unchanged. All credit for the original library goes to its authors and contributors.

## What the fork fixes (summary of [README.md](../README.md))

- No more `OverflowError: int too big to convert` with DeBERTa-v3 and other tokenizers that declare no max length. Such inputs are encoded without truncation, matching what the slow tokenizers did.
- Packaging moved to `pyproject.toml`.

The full details and links to the upstream issues are in [README.md](../README.md).

Pages: [Getting started](getting-started.md) · [Architecture](architecture.md) · [Contributing](contributing.md)

This repo has no CI workflow.

## License

MIT. Please keep citing the BERTScore paper. The original notice is kept: Copyright (c) 2019 Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi.
Modifications: Copyright (c) 2026 Joshua Almeida.
