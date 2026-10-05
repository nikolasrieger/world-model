import copy
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from ._dreamerv3 import networks, tools
from .adapters import check_action

to_np = lambda x: x.detach().cpu().numpy()

DEFAULT_CONFIG = {
    'dyn_stoch': 32,
    'dyn_deter': 512,
    'dyn_hidden': 512,
    'dyn_rec_depth': 1,
    'dyn_discrete': 32,
    'act': 'SiLU',
    'norm': True,
    'dyn_mean_act': 'none',
    'dyn_std_act': 'sigmoid2',
    'dyn_min_std': 0.1,
    'unimix_ratio': 0.01,
    'initial': 'learned',
    'num_actions': 6,
    'encoder': {'mlp_keys': '$^',
                'cnn_keys': 'image',
                'act': 'SiLU',
                'norm': True,
                'cnn_depth': 32,
                'kernel_size': 4,
                'minres': 4,
                'mlp_layers': 5,
                'mlp_units': 1024,
                'symlog_inputs': True},
    'decoder': {'mlp_keys': '$^',
                'cnn_keys': 'image',
                'act': 'SiLU',
                'norm': True,
                'cnn_depth': 32,
                'kernel_size': 4,
                'minres': 4,
                'mlp_layers': 5,
                'mlp_units': 1024,
                'cnn_sigmoid': False,
                'image_dist': 'mse',
                'vector_dist': 'symlog_mse',
                'outscale': 1.0},
    'reward_head': {'layers': 2, 'dist': 'symlog_disc', 'loss_scale': 1.0, 'outscale': 0.0},
    'cont_head': {'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0},
    'units': 512,
    'grad_heads': ['decoder', 'reward', 'cont'],
    'model_lr': 0.0001,
    'opt_eps': 1e-08,
    'grad_clip': 1000,
    'weight_decay': 0.0,
    'opt': 'adam',
    'precision': 32,
    'discount': 0.997,
    'kl_free': 1.0,
    'dyn_scale': 0.5,
    'rep_scale': 0.1,
    'batch_size': 16,
    'batch_length': 50
}


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update(
            {f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()}
        )
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = to_np(torch.mean(rep_loss))
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        assert "is_first" in obs
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


def load_checkpoint(path):
    # The original experiment saved torch.__version__ as a TorchVersion object.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location='cpu', weights_only=True)


def is_latent_checkpoint(ckpt):
    return ckpt.get('model_type') == 'latent' or (
        'config' in ckpt and 'dyn_stoch' in ckpt['config'] and 'weights' in ckpt)


class LatentModel(WorldModel):
    def __init__(self, config=None, device='cpu'):
        self.config = copy.deepcopy(DEFAULT_CONFIG if config is None else config)
        self.config['device'] = str(device)
        obs = SimpleNamespace(spaces={'image': SimpleNamespace(shape=(64, 64, 1))})
        super().__init__(obs, SimpleNamespace(n=6), 0, SimpleNamespace(**self.config))
        self.to(device)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        device = str(next(self.parameters()).device)
        self.config['device'] = self._config.device = device
        for module in self.modules():
            if hasattr(module, '_device'):
                module._device = device
        return result

    @property
    def decoder(self):
        return self.heads['decoder']

    @property
    def latent(self):
        return self.config['dyn_stoch'] * self.config['dyn_discrete']

    def _sample(self, state, noise=None, generator=None):
        dist = self.dynamics.get_dist(state)
        if noise is None and generator is None:
            return dist.sample()
        categorical = dist.base_dist
        probs = categorical.probs
        if noise is not None:
            expected = (len(probs), self.latent)
            if noise.shape != expected:
                raise ValueError(f'Expected Gumbel noise {expected}.')
            indices = (categorical.logits + noise.to(probs).reshape_as(probs)).argmax(-1)
        else:
            indices = torch.multinomial(probs.detach().cpu().flatten(0, 1), 1,
                                        generator=generator).reshape(probs.shape[:-1]).to(probs.device)
        return F.one_hot(indices, probs.shape[-1]).to(probs) + (probs - probs.detach())

    def prior(self, state, action, noise=None, generator=None):
        action = torch.as_tensor(action, device=state['deter'].device)
        check_action(action, len(state['deter']))
        controls = F.one_hot(action.long(), self.config['num_actions']).to(state['deter'])
        native_sample = noise is None and generator is None
        result = self.dynamics.img_step(state, controls, sample=native_sample)
        if not native_sample:
            result['stoch'] = self._sample(result, noise, generator)
        return result

    def _posterior_embed(self, deter, embed, noise=None, generator=None):
        hidden = self.dynamics._obs_out_layers(torch.cat((deter, embed), -1))
        result = dict(deter=deter, **self.dynamics._suff_stats_layer('obs', hidden))
        result['stoch'] = self._sample(result, noise, generator)
        return result

    def posterior(self, state, image, noise=None, generator=None):
        image = torch.as_tensor(image, device=state['deter'].device, dtype=torch.float32)
        if image.shape != (len(state['deter']), 1, 64, 64):
            raise ValueError('Expected image [batch, 1, 64, 64].')
        embed = self.encoder({'image': image.permute(0, 2, 3, 1)})
        return self._posterior_embed(state['deter'], embed, noise, generator)

    def initialize(self, context, noise=None, generator=None):
        device = next(self.parameters()).device
        images = torch.as_tensor(context['observations'], device=device, dtype=torch.float32)
        if images.ndim != 5 or images.shape[2:] != (1, 64, 64) or images.shape[1] < 1:
            raise ValueError('Expected pixel history [batch, time, 1, 64, 64].')
        batch, time = images.shape[:2]
        actions = torch.as_tensor(context.get('actions', torch.empty(batch, 0)), device=device)
        if actions.shape != (batch, time - 1):
            raise ValueError('Expected history actions [batch, time - 1].')
        check_action(actions.reshape(-1), batch * (time - 1))
        if noise is not None and noise.shape != (batch, time, self.latent):
            raise ValueError('Expected Gumbel noise [batch, time, stoch * classes].')
        embeds = self.encoder({'image': images.permute(0, 1, 3, 4, 2)})
        state = self.dynamics.initial(batch)
        for t in range(time):
            controls = images.new_zeros(batch, self.config['num_actions']) if t == 0 else F.one_hot(
                actions[:, t-1].long(), self.config['num_actions']).to(images)
            prior = self.dynamics.img_step(state, controls, sample=noise is None and generator is None)
            state = self._posterior_embed(prior['deter'], embeds[:, t],
                                          None if noise is None else noise[:, t], generator)
        return state

    def features(self, state):
        return self.dynamics.get_feat(state)

    def observe(self, state):
        image = self.decoder(self.features(state)[:, None])['image'].mode()
        return image[:, 0].permute(0, 3, 1, 2)

    def outcomes(self, state):
        feat = self.features(state)
        return dict(reward=self.heads['reward'](feat).mode().squeeze(-1),
                    termination_probability=1-self.heads['cont'](feat).mean.squeeze(-1))

    @classmethod
    def load(cls, ckpt, device='cpu'):
        config = ckpt.get('config', {})
        model = cls(config, device)
        model.load_state_dict(ckpt['weights'], strict=True)
        return model
