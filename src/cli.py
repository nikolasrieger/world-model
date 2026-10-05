from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch

from .helper import gif
from .latent_model import load_checkpoint
from .training import train


def prepare_config(cfg: DictConfig, output_dir: str) -> DictConfig:
    args = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key in ('data', 'output', 'resume', 'ckpt'):
        if args[key] is not None:
            args[key] = to_absolute_path(args[key])
    if args.command == 'gif':
        args.output = args.output or str(Path(output_dir) / 'prediction.gif')
        return args
    model = OmegaConf.to_container(args.model, resolve=True)
    name = model.pop('name')
    kind = 'latent' if name == 'dreamerv3' else 'mlp'
    ckpt = load_checkpoint(args.resume) if args.resume else {}
    if ckpt:
        if kind == 'latent':
            model = dict(ckpt['config'])
        else:
            model['hidden'] = ckpt['dimensions'][2]
            model['learning_rate'] = ckpt['opt']['param_groups'][0]['lr']
    args.model = kind
    args.model_config = model
    args.data = to_absolute_path(args.data or ckpt.get('data') or (
        'artifacts/pong-atari-pixels.pt' if kind == 'latent' else 'artifacts/pong-episodes.pt'))
    args.output = args.output or str(Path(output_dir) / 'checkpoint.pt')
    args.batch_size = args.batch_size if args.batch_size is not None else ckpt.get('batch_size', model['batch_size'])
    args.model_config.batch_size = args.batch_size
    if kind == 'latent':
        args.model_config.device = args.device
        if args.sequence_length is not None:
            args.model_config.batch_length = args.sequence_length
        if args.free_nats is not None:
            args.model_config.kl_free = args.free_nats
    return args


@hydra.main(version_base='1.3', config_path='configs', config_name='mlp')
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    args = prepare_config(cfg, output_dir)
    torch.set_num_threads(args.num_threads)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; use device=cpu.')
    OmegaConf.save(args, Path(output_dir) / 'effective-config.yaml', resolve=True)
    (gif if args.command == 'gif' else train)(args)


if __name__ == '__main__':
    main()
