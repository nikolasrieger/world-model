# world-model

Train Pong predictors from RAM (MLP) or grayscale images (DreamerV3).

The two example configs are [mlp.yaml](src/configs/mlp.yaml) and [dreamerv3.yaml](src/configs/dreamerv3.yaml).

```bash
uv run world-model --config-name mlp device=cpu
uv run world-model --config-name dreamerv3 device=mps
```

GIF `mode=one-step` supplies a real observation after every prediction; `mode=open-loop` predicts up to 200 steps without correction.

Both adapters support owned inference state through `WorldRuntime`:

```python
from src.adapters import MLPAdapter
from src.mlp_model import StateMLP
from src.runtime import WorldRuntime

runtime = WorldRuntime(MLPAdapter(StateMLP.load("artifacts/pong.pt", "cpu")))
observation = runtime.reset({"observations": normalized_ram_history})  # [batch, time, 128]
saved = runtime.snapshot()
child = runtime.branch()
prediction = child.step([1])  # one ALE action (0..5) per batch element
child.restore(saved)
```

Export two predicted branches and verify replay:

```bash
uv run python -m examples.branch_mlp --ckpt artifacts/pong.pt --device cpu --output artifacts/mlp-branches.npz
uv run python -m examples.branch_latent --ckpt PATH_TO_DREAMERV3_CHECKPOINT --device cpu --output artifacts/latent-branches.npz
uv run pytest
```