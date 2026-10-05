# world-model

Train Pong predictors from RAM (`mlp`) or grayscale images (`latent`).

Both models use the shared loop in `src/training.py`. `MLPTraining` and
`LatentTraining` supply their own data preparation, training step, evaluation, and
checkpoint fields; the loop handles batching, stopping, logging, and saving.
`src/cli.py` handles command-line arguments.

The RAM predictor is a small deterministic network predicting the next RAM state.
`src/latent_model.py` contains the **NM512 DreamerV3 world model used for the 25,000-update
Pong run**: CNN encoder/decoder, categorical RSSM, reward and continuation heads.
Its network/distribution primitives live in `src/_dreamerv3` with the upstream MIT
license and [source provenance](src/_dreamerv3/NOTICE.md). This integrates the pinned
2023 implementation; it does not include Dreamer's actor/critic or policy training.

Use the already-trained checkpoint directly:

```bash
uv run world-model gif --ckpt artifacts/reference-control/run-25000/checkpoint.pt --output artifacts/pong-dreamerv3.gif --mode one-step --device mps
uv run world-model train --resume artifacts/reference-control/run-25000/checkpoint.pt --output artifacts/pong-dreamerv3.pt --max-updates 1000 --device mps
```

`--max-updates` counts **additional** optimizer updates, so the second command runs
from 25,000 to 26,000. The separate output path preserves the original checkpoint.
Use `--device cpu` on machines without MPS.

To train from scratch or resume the new local output:

```bash
uv run world-model train --model latent --device mps
uv run world-model train --resume artifacts/pong-dreamerv3.pt --device mps
```

Fresh latent training defaults: batch size 16, sequence length 50, evaluation context
5 and horizon 5, Adam learning rate `1e-4`, epsilon `1e-8`, gradient clip 1000.
The native objective uses summed image squared error, symlog two-hot reward loss,
continuation loss, and KL weights 0.5/0.1 with 1 free nat. Training batches contain
50 frames and 49 incoming actions; their first action/reward are zero-padded and
`is_first` is set. Windows never cross episode resets.

Resume restores architecture, optimizer, CPU/MPS sampling RNG, shuffled window order
and cursor. Sequence length, free nats, and batch size come from the checkpoint unless
explicitly overridden. Changing the sequence length or appending data starts a new
window permutation; switching devices does not reproduce the same stochastic trajectory.
`--history` and `--horizon` control validation, not training sequence length.

For latent training, evaluation saves the latest state to the output path and the
best validation pixel-MSE state **within this invocation** to `<name>.best.pt`.
MLP training retains its best-only checkpoint policy and evaluates after each epoch
or an explicit update limit. `--max-updates` now applies to both models. MLP keeps
partial final batches; Dreamer uses full batches only.
The previous local RSSM architecture has been removed; its old checkpoints are not
compatible and produce an explicit loading error. Existing artifact files are preserved.

GIF `one-step` supplies a real observation after each prediction; `open-loop` predicts
up to 200 steps without correction. Integrating the checkpoint preserves its learned
behavior; it does not improve its long-horizon prediction quality.

```bash
uv run pytest -q
```
