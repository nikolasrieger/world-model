# world-model

Train Pong predictors from RAM (`mlp`) or grayscale images (`latent`).

The RAM predictor is a small deterministic network predicting the next RAM state. The image predictor is a discrete recurrent state-space model (RSSM), based on the DreamerV2. 

```bash
uv run world-model train --model latent --device mps
uv run world-model train --resume artifacts/pong-rssm.pt --device mps
```

RSSM defaults are batch size 16, sequence length 50, evaluation context 5 and horizon 5, learning rate 2e-4. `--sequence-length` controls training; `--history` and `--horizon` control validation and the checkpoint's default visualization context. Windows never cross episode resets. 

Training saves the best validation pixel-MSE checkpoint. Resume retains architecture, cache, output path, sequence length, and all other states and variables. 

```bash
uv run world-model gif --ckpt artifacts/pong-rssm.pt --output artifacts/pong-rssm.gif --mode one-step --device mps
```

GIF `one-step` supplies a real observation after each prediction; `open-loop` instead predicts up to 200 steps without correction.