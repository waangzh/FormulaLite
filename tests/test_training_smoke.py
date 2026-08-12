from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from formulalite.data import FormulaDataConfig, FormulaDataModule, FormulaTokenizer
from formulalite.model import FormulaLiteConfig, FormulaLiteForImageToLatex
from formulalite.training import FormulaLiteLitModule

from test_training import training_config


def _module(config: FormulaLiteConfig, tokenizer: FormulaTokenizer) -> FormulaLiteLitModule:
    return FormulaLiteLitModule(
        FormulaLiteForImageToLatex(config),
        tokenizer,
        training_config(),
    )


def _trainer(root: Path, *, max_epochs: int) -> L.Trainer:
    checkpoint = ModelCheckpoint(
        dirpath=root / "checkpoints",
        filename="best",
        monitor="BLEU",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=max_epochs,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(root, name="logs"),
        callbacks=[checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )


def test_tiny_trainer_fit_validate_checkpoint_and_resume(
    tiny_model_config: FormulaLiteConfig,
    tmp_path: Path,
) -> None:
    manifest = Path(__file__).parent / "fixtures" / "dataset" / "manifest.jsonl"
    tokenizer = FormulaTokenizer.build()
    data = FormulaDataModule(
        FormulaDataConfig(
            train_manifest=manifest,
            val_manifest=manifest,
            batch_size=2,
            num_workers=0,
            max_sequence_length=32,
            unknown_policy="drop",
            overlength_policy="drop",
            image_size=64,
        ),
        tokenizer=tokenizer,
    )

    trainer = _trainer(tmp_path / "first", max_epochs=1)
    trainer.fit(_module(tiny_model_config, tokenizer), datamodule=data)
    validation = trainer.validate(_module(tiny_model_config, tokenizer), datamodule=data)
    checkpoint_path = tmp_path / "first" / "checkpoints" / "last.ckpt"

    assert trainer.global_step == 1
    assert validation[0].keys() >= {"val_loss", "BLEU", "edit_distance"}
    assert checkpoint_path.is_file()

    resumed = _trainer(tmp_path / "first", max_epochs=2)
    resumed.fit(
        _module(tiny_model_config, tokenizer),
        datamodule=data,
        ckpt_path=checkpoint_path,
    )
    assert resumed.global_step == 2
