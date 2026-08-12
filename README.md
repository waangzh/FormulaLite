# FormulaLite

FormulaLite is a lightweight image-to-LaTeX OCR system designed for local and browser
inference.

FormulaLite currently supports:

- compact 687-token LaTeX tokenizer
- deterministic cross-runtime preprocessing
- map and streaming datasets
- HGNetV2 vision encoder
- compact MBART decoder
- complete image-to-LaTeX encoder-decoder model
- resumable Lightning training with TensorBoard metrics

No trained model or benchmark results are distributed yet.

## Development

FormulaLite requires Python 3.11 or newer and uses a standard `src` package layout.

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run pyright
```

## Training

Set the dataset manifest paths in the Hydra data config, then start a baseline run:

```bash
uv run python -m formulalite.train
```

Initialize from a converted FormulaLite `save_pretrained()` directory with:

```bash
uv run python -m formulalite.train \
  training.init_mode=pretrained \
  training.pretrained_path=/path/to/model
```

Resume model, optimizer, scheduler, epoch, and global-step state from a Lightning
checkpoint with:

```bash
uv run python -m formulalite.train \
  training.resume_from_checkpoint=/path/to/last.ckpt
```

The default GPU precision is `bf16-mixed`; use `trainer.precision=32-true` where bf16 is
unavailable. Inspect losses, validation metrics, epoch, and learning rate with:

```bash
uv run tensorboard --logdir outputs
```
