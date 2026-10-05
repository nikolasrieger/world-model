"""Packaged Hydra examples and model defaults."""
from importlib.resources import files
from omegaconf import DictConfig, OmegaConf


def load_model_config(name: str) -> DictConfig:
    with files(__package__).joinpath('model', f'{name}.yaml').open() as stream:
        config = OmegaConf.load(stream)
    del config['name']
    return config
