import numpy as np
import torch
from PIL import Image, ImageDraw
from .atari import AtariRAMEnv, RAM_CONFIG, environment_spec

CONFIG = RAM_CONFIG
DEFAULT_HORIZON = 10000


class PongEnv(AtariRAMEnv):
    def __init__(self, render_mode=None, max_steps=DEFAULT_HORIZON):
        super().__init__(environment_spec(), render_mode, max_steps)


def ram_image(state):
    values = np.asarray(state)
    if values.shape != (128,) or not np.isfinite(values).all():
        raise ValueError('Expected 128 finite normalized Pong RAM values.')
    ram = np.rint(np.clip(values, 0, 1) * 255).astype(int)
    image = Image.new('RGB', (160, 210), (12, 18, 28))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 34, 159, 193), outline=(50, 60, 75))
    for x, y, color in [(16, ram[50] - 15, (213, 130, 74)),
                         (140, ram[51] - 13, (92, 186, 92))]:
        top, bottom = max(34, y), min(193, y + 14)
        if bottom >= top:
            draw.rectangle((x, top, x + 3, bottom), fill=color)
    if ram[54] and ram[49] > 49:
        x, y = ram[49] - 49, ram[54] - 14
        draw.rectangle((x, y, x + 1, y + 3), fill='white')
    return image


def ram_metrics(prediction, target):
    pred, true = prediction * 255, target * 255
    result = {'paddle_y_mae': float((pred[..., [50, 51]] - true[..., [50, 51]]).abs().mean())}
    visible = (true[..., 49] > 49) & (true[..., 49] < 209) & (true[..., 54] >= 48) & (true[..., 54] <= 207)
    pred_visible = (pred[..., 49] > 49) & (pred[..., 49] < 209) & (pred[..., 54] >= 48) & (pred[..., 54] <= 207)
    result['ball_position_error'] = float(torch.linalg.vector_norm(
        pred[..., [49, 54]] - true[..., [49, 54]], dim=-1)[visible].mean()) if visible.any() else None
    result['ball_visible_recall'] = float(pred_visible[visible].float().mean()) if visible.any() else None
    result['ball_visible_count'] = int(visible.sum())
    return result


class RallyReadiness:
    def __init__(self):
        self.previous = None
        self.moving = False

    def __call__(self, env):
        base = env if hasattr(env, 'unwrapped') else env.env
        ram = base.unwrapped.ale.getRAM().astype(int)
        left, right = ram[50] - 15, ram[51] - 13
        ball = (ram[49] - 49, ram[54] - 14)
        visible = (20 <= left <= 193 and 20 <= right <= 193
                   and 0 < ball[0] < 160 and 34 <= ball[1] <= 193)
        if not visible:
            self.previous, self.moving = None, False
            return False
        self.moving |= self.previous is not None and ball != self.previous
        self.previous = ball
        return self.moving
