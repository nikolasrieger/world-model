import torch
from torch import nn
from .configs import load_model_config
from torch.nn import functional as F


class StateMLP(nn.Module):
    def __init__(self, hidden=None, observation_dim=None, num_actions=None):
        super().__init__()
        config = load_model_config('mlp')
        hidden = config.hidden if hidden is None else hidden
        observation_dim = config.observation_shape[0] if observation_dim is None else observation_dim
        num_actions = config.num_actions if num_actions is None else num_actions
        self.observation_dim, self.num_actions = observation_dim, num_actions
        self.observation_shape = (observation_dim,)
        self.net = nn.Sequential(
            nn.Linear(observation_dim + num_actions, hidden),
            nn.Tanh(), 
            nn.Linear(hidden, hidden),
            nn.Tanh(), 
            nn.Linear(hidden, observation_dim))
        for name in ("state", "delta"):
            self.register_buffer(name + "_mean", torch.zeros(observation_dim))
            self.register_buffer(name + "_scale", torch.ones(observation_dim))

    def forward(self, state, action):
        inputs = torch.cat(((state - self.state_mean) / self.state_scale, F.one_hot(action.long(), self.num_actions)), -1)
        return state + self.delta_mean + self.delta_scale * self.net(inputs)

    @torch.no_grad()
    def normalize(self, states, targets):
        for name, values in (("state", states), ("delta", targets - states)):
            getattr(self, name + "_mean").copy_(values.mean(0))
            getattr(self, name + "_scale").copy_(values.std(0, unbiased=False).clamp_min(1e-3))

    @classmethod
    def load(cls, path, device):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        observation_dim, num_actions, hidden = ckpt["dimensions"]
        model = cls(hidden, observation_dim, num_actions).to(device)
        weights = {k: v for k, v in ckpt["weights"].items() if not k.startswith("outcome_net.")}
        model.load_state_dict(weights)
        return model.eval()
