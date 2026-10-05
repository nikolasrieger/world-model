import argparse
import torch
from .helper import gif
from .training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--model", choices=("mlp", "latent"), default="mlp")
    fit.add_argument("--output")
    fit.add_argument("--data")
    fit.add_argument("--history", type=int, default=5)
    fit.add_argument("--sequence-length", type=int)
    fit.add_argument("--free-nats", type=float)
    fit.add_argument("--eval-every", type=int, default=250)
    fit.add_argument("--max-updates", type=int)
    fit.add_argument("--val-windows", type=int, default=128)
    fit.add_argument("--samples", type=int, default=100000)
    fit.add_argument("--resume")
    fit.add_argument("--collect-more", type=int, default=0, metavar="N")
    fit.add_argument("--horizon", type=int, default=5)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--batch-size", type=int)
    render = commands.add_parser("gif")
    render.add_argument("--ckpt")
    render.add_argument("--output")
    render.add_argument("--mode", choices=("open-loop", "one-step"), default="open-loop")
    render.add_argument("--history", type=int, default=5)
    for command in (fit, render):
        command.add_argument("--device", choices=("cpu", "mps"), default="cpu")
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu.")
    if args.command == "gif":
        gif(args)
        return
    ckpt = {}
    if args.resume:
        from .latent_model import load_checkpoint, is_latent_checkpoint
        ckpt = load_checkpoint(args.resume)
        kind = "latent" if is_latent_checkpoint(ckpt) else "mlp"
        args.model = kind
    latent = args.model == "latent"
    args.data = args.data or ckpt.get("data") or ("artifacts/pong-atari-pixels.pt" if latent else "artifacts/pong-episodes.pt")
    args.output = args.output or args.resume or ("artifacts/pong-dreamerv3.pt" if latent else "artifacts/pong.pt")
    if args.batch_size is None:
        args.batch_size = ckpt.get("config", {}).get("batch_size", 16) if latent else 256
    if min(args.history, args.horizon, args.epochs, args.batch_size, args.samples) < 1 or args.collect_more < 0:
        parser.error("Training sizes must be positive; --collect-more must be nonnegative.")
    if args.eval_every < 1 or args.val_windows < 1 or (args.max_updates is not None and args.max_updates < 1):
        parser.error("Evaluation interval, validation windows, and max updates must be positive.")
    train(args)


if __name__ == "__main__":
    main()
