from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).parents[1] / "config"


def compose_config(*overrides: str) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="train", overrides=list(overrides))


def test_omegaconf_loads_root_config() -> None:
    config = OmegaConf.load(CONFIG_DIR / "train.yaml")

    assert config.project_name == "formulalite"
    assert config.defaults


def test_hydra_composes_baseline_config() -> None:
    config = compose_config()

    assert config.model.input_size == 384
    assert config.model.encoder_hidden_size == 2048
    assert config.model.decoder_hidden_size == 384
    assert config.model.decoder_layers == 2
    assert config.model.decoder_attention_heads == 16
    assert config.model.decoder_ffn_dim == 1536
    assert config.model.vocab_size == 687


def test_hydra_composes_tiny_config() -> None:
    config = compose_config("data=tiny", "model=tiny", "trainer=tiny")

    assert config.data.name == "tiny"
    assert config.data.batch_size == 2
    assert config.data.image_size == 64
    assert config.model.name == "tiny"
    assert config.model.input_size < 384
    assert config.model.encoder_hidden_size < 2048
    assert config.model.decoder_hidden_size < 384
    assert config.trainer.max_epochs == 1
    assert config.trainer.precision == "32-true"
