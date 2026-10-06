from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch

from .helper import gif
from .latent_model import load_checkpoint
from .training import train
from .envs.atari import environment_spec, default_data_path, make_env


def prepare_config(cfg: DictConfig, output_dir: str) -> DictConfig:
    args = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key in ('data', 'output', 'resume', 'ckpt'):
        if args[key] is not None:
            args[key] = to_absolute_path(args[key])
    checkpoint_path = args.ckpt if args.command in ('gif', 'diagnose') else args.resume
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path else {}
    args.environment = environment_spec(args, checkpoint)
    if args.command in ('gif', 'diagnose'):
        filename = 'prediction.gif' if args.command == 'gif' else 'diagnostics.json'
        args.output = args.output or str(Path(output_dir) / filename)
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
    env = make_env(args.environment, kind == 'latent')
    try:
        num_actions = int(env.action_space.n)
        observation_shape = ([1, args.environment.image_size, args.environment.image_size]
                             if kind == 'latent' else list(env.observation_space.shape))
    finally:
        env.close()
    if ckpt:
        old_actions = ckpt['config']['num_actions'] if kind == 'latent' else ckpt['dimensions'][1]
        old_shape = ckpt['config'].get('observation_shape', [1, 64, 64]) if kind == 'latent' else [ckpt['dimensions'][0]]
        if num_actions != old_actions or list(old_shape) != observation_shape:
            raise ValueError('Checkpoint dimensions do not match the environment.')
    model['num_actions'] = num_actions
    model['observation_shape'] = observation_shape
    args.model_config = model
    args.data = to_absolute_path(args.data or ckpt.get('data') or default_data_path(args.environment, kind == 'latent'))
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
    from .diagnostics import diagnose
    commands = {'train': train, 'gif': gif, 'diagnose': diagnose}
    if args.command not in commands:
        raise ValueError(f'Unknown command: {args.command}')
    commands[args.command](args)


if __name__ == '__main__':
    main()
