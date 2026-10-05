# world-model

Train Pong predictors from RAM (MLP) or grayscale images (DreamerV3).

The two example configs are [mlp.yaml](src/configs/mlp.yaml) and [dreamerv3.yaml](src/configs/dreamerv3.yaml).

```bash
uv run world-model --config-name mlp device=cpu
uv run world-model --config-name dreamerv3 device=mps
```

GIF `mode=one-step` supplies a real observation after every prediction; `mode=open-loop` predicts up to 200 steps without correction.
