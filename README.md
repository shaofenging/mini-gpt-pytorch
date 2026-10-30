# MiniGPT — Decoder-only Transformer in PyTorch

## Overview

MiniGPT is a small decoder-only Transformer language model implemented in PyTorch. It uses a character-level tokenizer and the Tiny Shakespeare dataset, and supports training, validation, checkpoint saving/loading, and autoregressive text generation.

This version builds on [HaoPan036/Mini-GPT](https://github.com/HaoPan036/Mini-GPT). See [Attribution](#attribution) for the upstream source and changes in this version.

## Architecture

```text
Token Embedding + Learned Position Embedding
                  ↓
Transformer Block × n_layer
  Pre-LayerNorm → Causal Multi-Head Self-Attention → Residual Connection
  Pre-LayerNorm → Feed-Forward Network            → Residual Connection
                  ↓
Final LayerNorm → Language Model Head → Next-token Logits
```

The feed-forward network uses `n_embd → 4 * n_embd → n_embd` with ReLU and dropout. Autoregressive generation samples one token at a time and appends it to the context.

## Model Configuration

Defaults in `config.py`:

| Parameter | Value |
| --- | --- |
| `n_embd` | 384 |
| `num_heads` | 6 |
| `head_size` | 64 (`n_embd // num_heads`) |
| `n_layer` | 6 |
| `block_size` | 256 |
| `dropout` | 0.2 |
| Character vocabulary size | Derived from `data/input.txt` |

## Attention Implementation

Each head computes standard Scaled Dot-Product Attention:

```text
softmax(QK^T / sqrt(d_k) + causal_mask) V
```

`d_k` is the Q/K dimension of a single attention head, equal to `head_size`. The implementation obtains it directly from `k.size(-1)`. The causal mask prevents a position from attending to future tokens. Head outputs are concatenated and passed through an output projection.

## Installation

Python 3.12 was used for local verification. Run the following commands from the project root:

```bash
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

```bash
python train.py
```

Training reads `data/input.txt`, builds the character vocabulary, and splits the token sequence into 90% training and 10% validation data. The included data is Tiny Shakespeare; a dataset reference is available in [karpathy/char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).

Edit `config.py` to change the model and training configuration. The current defaults run 20 optimizer updates as a smoke-training configuration, with `batch_size=64`, `learning_rate=3e-4`, and `seed=42`. CUDA is selected when available; otherwise training uses CPU. These defaults check the pipeline and do not establish model quality.

Python, NumPy, and PyTorch are seeded before model initialization and batch sampling. Reproducibility across different devices or library versions is not guaranteed.

Logged steps count completed optimizer updates. Train/validation loss is evaluated initially, periodically, and after the final update. The final evaluation corresponds to the parameters saved in `ckpt.pt`. Training overwrites that file and then generates a sample under `torch.no_grad()`.

Checkpoints are generated locally and are not included in the repository. Earlier checkpoints trained with embedding-dimension attention scaling should be retrained for the current implementation.

## Text Generation

Train a model first to create `ckpt.pt`, then run:

```bash
python generate.py
```

Generate from a prompt:

```bash
python generate.py --prompt "ROMEO:" --max_new_tokens 200
```

Control sampling:

```bash
python generate.py --prompt "ROMEO:" --temperature 0.8 --top_k 20 --checkpoint ckpt.pt --device cpu
```

Prompts must contain characters in the saved tokenizer vocabulary. `temperature` must be positive, and `top_k` must be a positive integer when supplied. Generation uses the last `block_size` tokens as context. Run `python generate.py --help` for all supported arguments.

## Tests

```bash
python -m pytest
```

The 10 core tests cover:

- Tokenizer encode/decode round trip
- Attention output shape
- Multi-Head Attention output shape
- Causal masking in attention and GPT
- GPT forward logits and finite scalar loss
- Checkpoint save/load with identical logits and tokenizer mappings
- Numerical Attention correctness against an explicit scaled dot-product calculation
- Input validation for head configuration, sequence length, and sampling parameters

The numerical test uses `n_embd=12` and `head_size=4` to distinguish head-dimension scaling from embedding-dimension scaling. Checkpoint tests use temporary files.

## Project Structure

```text
mini-gpt-pytorch/
├── data/
│   └── input.txt
├── tests/
│   └── test_core.py
├── model.py
├── train.py
├── generate.py
├── utils.py
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Project Scope

This is a small language model and GPT architecture implementation focused on understanding and validating Transformer internals. It is not an instruction-tuned assistant or a question-answering model, and does not provide ChatGPT-level capabilities.

## Attribution

This project is based on [HaoPan036/Mini-GPT](https://github.com/HaoPan036/Mini-GPT), which was used as the starting point for learning, cleanup, and modification. The upstream implementation remains the basis of the model and training pipeline.

Changes in this version include:

- Standard attention scaling correction using the per-head Q/K dimension
- Reproducibility improvements through shared random seed configuration
- Final evaluation/checkpoint alignment
- Input validation checks
- Core correctness tests
- Project cleanup and documentation

The upstream project credits Andrej Karpathy's nanoGPT. References retained from the original project:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

The upstream README mentions MIT, but no standalone `LICENSE` file is included in this checkout. This cleanup does not add or assign a new license. Upstream licensing terms should be clarified before public redistribution.
