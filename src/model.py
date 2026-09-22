import torch
from torch import nn
from torch.nn import functional as F


class StateMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(134, hidden), 
            nn.Tanh(), 
            nn.Linear(hidden, hidden),
            nn.Tanh(), 
            nn.Linear(hidden, 128))
        for name in ("state", "delta"):
            self.register_buffer(name + "_mean", torch.zeros(128))
            self.register_buffer(name + "_scale", torch.ones(128))

    def forward(self, state, action):
        inputs = torch.cat(((state - self.state_mean) / self.state_scale, F.one_hot(action.long(), 6)), -1)
        return state + self.delta_mean + self.delta_scale * self.net(inputs)

    @torch.no_grad()
    def normalize(self, states, targets):
        for name, values in (("state", states), ("delta", targets - states)):
            getattr(self, name + "_mean").copy_(values.mean(0))
            getattr(self, name + "_scale").copy_(values.std(0, unbiased=False).clamp_min(1e-3))

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        _, _, hidden = checkpoint["dimensions"]
        model = cls(hidden).to(device)
        weights = {k: v for k, v in checkpoint["weights"].items() if not k.startswith("outcome_net.")}
        model.load_state_dict(weights)
        return model.eval()
