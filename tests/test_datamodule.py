from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from formulalite.data.datamodule import FormulaDataConfig, FormulaDataModule

ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "config"
MANIFEST = ROOT / "tests" / "fixtures" / "dataset" / "manifest.jsonl"


def data_config(seed: int = 42) -> FormulaDataConfig:
    return FormulaDataConfig(
        name="tiny",
        train_manifest=MANIFEST,
        val_manifest=MANIFEST,
        test_manifest=MANIFEST,
        batch_size=2,
        num_workers=0,
        max_sequence_length=32,
        unknown_policy="drop",
        overlength_policy="drop",
        shuffle=True,
        seed=seed,
    )


def order_for_seed(seed: int) -> list[str]:
    module = FormulaDataModule(data_config(seed))
    module.setup("fit")
    return [sample_id for batch in module.train_dataloader() for sample_id in batch.sample_ids]


def test_datamodule_setup_and_loader_smoke() -> None:
    module = FormulaDataModule(data_config())
    module.setup("fit")
    batch = next(iter(module.train_dataloader()))
    assert batch.pixel_values.shape == (2, 3, 384, 384)
    assert batch.decoder_input_ids.shape == batch.labels.shape
    assert len(batch.subsets) == 2
    module.setup("test")
    test_batch = next(iter(module.test_dataloader()))
    assert all(subset in {"SPE", "CPE", "SCE", "HWE"} for subset in test_batch.subsets)


def test_map_shuffle_reproducibility() -> None:
    assert order_for_seed(11) == order_for_seed(11)
    assert order_for_seed(11) != order_for_seed(12)


def test_tiny_config_composes_and_constructs_datamodule() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="train", overrides=["data=tiny"])
    value = OmegaConf.to_container(config.data, resolve=True)
    assert isinstance(value, dict)
    data = {str(key): item for key, item in value.items()}
    data["train_manifest"] = MANIFEST
    data["val_manifest"] = MANIFEST
    data["test_manifest"] = MANIFEST
    module = FormulaDataModule(data)
    module.setup("fit")
    assert next(iter(module.val_dataloader())).pixel_values.shape[1:] == (3, 64, 64)
