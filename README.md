# world-model

Train Pong predictors from RAM (`mlp`) or grayscale images (`latent`).

- **RAM predictor (`mlp`)**: a small deterministic network that takes the game's
  128 RAM values and an action, then predicts the next RAM state. Multi-step
  predictions feed each predicted state back into the network.
- **Image predictor (`latent`)**: a recurrent stochastic model trained on 64×64
  grayscale frames. It encodes observed frames into a latent state, predicts how
  that state changes with actions, and decodes future images. It also predicts
  rewards and episode termination. The default spatial architecture keeps the
  sampled latent state on a 32×32 grid.

Both learn from episodes collected with random actions. They predict what
happens under supplied actions; they do not learn a policy for playing Pong.

```bash
uv run world-model train --model latent --device mps
uv run world-model train --resume artifacts/pong-latent.pt --device mps
```

Use `--device cpu` without Apple GPU support. Episodes are cached in `artifacts/`;
`--collect-more N` appends data. Training saves the best validation checkpoint.

For the latent model, `--history H` sets the observed context. `--horizon K`
sets the prediction length for either model. Windows never cross episode resets. Latent defaults are
4 context frames, 5 predicted frames, batch size 32, and 30 epochs. Adjust with
`--batch-size` and `--epochs`; progress reports batch timing and training ETA.

Generate a prediction GIF:

```bash
uv run world-model gif --ckpt artifacts/pong-latent.pt --output artifacts/pong.gif
```
