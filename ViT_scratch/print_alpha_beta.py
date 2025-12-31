import torch
from utils import load_experiment
import argparse


def print_alpha_beta(model):
    for name, module in model.named_modules():
        # RGB_Depth_Agreement_Refined: alpha_raw/beta_raw
        if hasattr(module, "get_alpha_beta"):
            try:
                a, b = module.get_alpha_beta()
                a_val = a.item() if torch.is_tensor(a) else float(a)
                b_val = b.item() if torch.is_tensor(b) else float(b)
                print(f"{name}: alpha={a_val:.6f}, beta={b_val:.6f} (via get_alpha_beta)")
            except Exception:
                if getattr(module, "alpha_raw", None) is not None:
                    a = torch.sigmoid(module.alpha_raw)
                    b = torch.sigmoid(module.beta_raw)
                    print(f"{name}: alpha={a.item():.6f}, beta={b.item():.6f} (sigmoid alpha_raw/beta_raw)")
                else:
                    print(f"{name}: get_alpha_beta() failed")

        # modules with alpha and beta fixed in config
        elif hasattr(module, "alpha") and hasattr(module, "beta"):
            try:
                print(f"{name}: alpha={module.alpha:.6f}, beta={module.beta:.6f} (fixed)")
            except Exception:
                print(f"{name}: alpha={module.alpha}, beta={module.beta} (fixed)")

        # modules with raw params only
        elif getattr(module, "alpha_raw", None) is not None and getattr(module, "beta_raw", None) is not None:
            a = torch.sigmoid(module.alpha_raw)
            b = torch.sigmoid(module.beta_raw)
            print(f"{name}: alpha={a.item():.6f}, beta={b.item():.6f} (sigmoid alpha_raw/beta_raw)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", required=True, help="experiment name used by load_experiment")
    parser.add_argument("--checkpoint", "-c", default="model_final.pt")
    parser.add_argument("--depth", action="store_true", help="pass depth=True to load_experiment")
    parser.add_argument("--map_location", default="cpu")
    parser.add_argument("--override_learnable", action="store_true", help="Set override_config={'learnable_alpha_beta': True}")
    args = parser.parse_args()

    override_config = None
    if args.override_learnable:
        override_config = {"learnable_alpha_beta": True}

    config, model, _, _, _, _ = load_experiment(
        args.experiment,
        checkpoint_name=args.checkpoint,
        depth=args.depth,
        map_location=args.map_location,
        override_config=override_config,
    )

    model = model.to(args.map_location)
    model.eval()

    print(f"Loaded experiment: {args.experiment} | learnable override: {override_config}")
    print_alpha_beta(model)
