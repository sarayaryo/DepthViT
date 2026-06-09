"""
Inference cost logger for all fusion experiments.

Usage:
    # Quick test with 200 samples, 1 experiment only
    python test_for_inferencelog.py --max_samples 200 --only NYU_latefusion_lr1e-3

    # Full run (all 14 experiments, full test set)
    python test_for_inferencelog.py

    # Adjusted-lr experiments only, 500 samples
    python test_for_inferencelog.py --max_samples 500 --group adjusted

Output:
    experiments/<exp_name>/inference.log
"""

import argparse
import gc
import logging
import os
import time

import psutil
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from data import load_datapath_NYU, get_dataloader
from train import Trainer, process_attention_data
from train_CMI import ConditionalCLUB, process_attention_data_forCMI
from utils import load_experiment


# ── experiment definitions ──────────────────────────────────────────

EXPERIMENTS = [
    # Experiment 1 & 4 : default lr  (_lr1e-3)
    {"base": "NYU_latefusion",              "sched": True,  "learn": False, "ckpt": "model_final.pt"},
    {"base": "NYU_sharefusion_a0.0_b0.5",   "sched": True,  "learn": False, "ckpt": "model_final.pt"},
    {"base": "NYU_sharefusion_a0.5_b0.0",   "sched": True,  "learn": False, "ckpt": "model_final.pt"},
    {"base": "NYU_sharefusion_a0.5_b0.5",   "sched": True,  "learn": False, "ckpt": "model_final.pt"},
    {"base": "NYU_sharefusion_a0.25_b0.25", "sched": True,  "learn": False, "ckpt": "model_final.pt"},
    {"base": "NYU_sharefusion_alearn_blearn","sched": True,  "learn": True,  "ckpt": "model_final.pt",
     "dir_override": "NYU_sharefusion_alearn_blearn_lr1e-3_revise"},
    {"base": "NYU_ARfusion_alearn_blearn",  "sched": True,  "learn": True,  "ckpt": "model_16.pt"},

    # Experiment 2 & 5 : adjusted lr  (no suffix)
    {"base": "NYU_latefusion",              "sched": False, "learn": False, "ckpt": "model_final.pt"},
    {"base": "NYU_sharefusion_a0.0_b0.5",   "sched": False, "learn": False, "ckpt": "model_final.pt"},
    {"base": "NYU_sharefusion_a0.5_b0.0",   "sched": False, "learn": False, "ckpt": "model_final.pt"},
    {"base": "NYU_sharefusion_a0.5_b0.5",   "sched": False, "learn": False, "ckpt": "model_final.pt"},
    {"base": "NYU_sharefusion_a0.25_b0.25", "sched": False, "learn": False, "ckpt": "model_final.pt"},
    {"base": "NYU_sharefusion_alearn_blearn","sched": False, "learn": True,  "ckpt": "model_final.pt"},
    {"base": "NYU_ARfusion_alearn_blearn",  "sched": False, "learn": True,  "ckpt": "model_final.pt"},
]


def exp_dir_name(cfg):
    if "dir_override" in cfg:
        return cfg["dir_override"]
    return cfg["base"] + ("_lr1e-3" if cfg["sched"] else "")


# ── logging helpers ─────────────────────────────────────────────────

def make_logger(path):
    name = f"infer.{path}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return logger


def log_mem(logger, tag):
    vm = psutil.virtual_memory()
    logger.info(f"{tag}: CPU {vm.percent:.1f}%  |  "
                f"Used {vm.used / 2**30:.2f} GB / Total {vm.total / 2**30:.2f} GB  |  "
                f"Avail {vm.available / 2**30:.2f} GB")
    if torch.cuda.is_available():
        d = torch.cuda.current_device()
        a = torch.cuda.memory_allocated(d) / 2**20
        r = torch.cuda.memory_reserved(d) / 2**20
        logger.info(f"{tag}: GPU allocated {a:.2f} MB  |  reserved {r:.2f} MB")
    else:
        logger.info(f"{tag}: GPU not available (CPU mode)")


# ── timed inference ─────────────────────────────────────────────────

@torch.no_grad()
def timed_inference(model, test_loader, loss_fn, device, logger, exp_label=""):
    model.eval()
    n_samples = len(test_loader.dataset)
    n_batches = len(test_loader)

    log_mem(logger, "before_inference")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    total_loss = 0
    correct = 0
    attn_data = []
    cls_pairs = []

    pbar = tqdm(test_loader, desc=f"  {exp_label}", unit="batch",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for idx, batch in enumerate(pbar):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bt0 = time.perf_counter()

        images = batch["image"].to(device)
        depth  = batch["depth"].to(device)
        labels = batch["label"].to(device)
        paths  = batch["path"]

        logits, out_img, out_dpt, att_img, att_dpt = model(
            images, depth, attentions_choice=True
        )
        cls_pairs.append((out_img[:, 0, :], out_dpt[:, 0, :]))

        n_layers = len(att_img)
        attn_data.extend(
            process_attention_data(images, depth, labels, att_img, att_dpt, n_layers - 1, image_paths=paths)
        )

        loss = loss_fn(logits, labels)
        total_loss += loss.item() * len(images)
        correct += (logits.argmax(1) == labels).sum().item()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bt1 = time.perf_counter()

        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/((idx+1)*images.size(0)):.3f}")

        if idx % 100 == 0:
            log_mem(logger, f"batch {idx}/{n_batches}")
            logger.info(f"  elapsed {bt1 - t0:.2f}s  |  this batch {(bt1 - bt0)*1000:.1f}ms")

        del batch, images, depth, labels, logits, loss, att_img, att_dpt
        gc.collect()
        torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    log_mem(logger, "after_inference")

    acc = correct / n_samples
    avg_loss = total_loss / n_samples

    logger.info("--- Results ---")
    logger.info(f"Total time       : {total_time:.4f} s")
    logger.info(f"Per-sample time  : {total_time / n_samples * 1000:.4f} ms")
    logger.info(f"Per-batch time   : {total_time / n_batches * 1000:.4f} ms")
    logger.info(f"Throughput       : {n_samples / total_time:.2f} samples/s")
    logger.info(f"Accuracy         : {acc:.4f}")
    logger.info(f"Average loss     : {avg_loss:.4f}")

    return acc, avg_loss, total_time, attn_data


# ── main ────────────────────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    base_path = r"../data/nyu_data/nyu2"

    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    image_paths, depth_paths, _ = load_datapath_NYU(base_path, args.seed)

    if args.max_samples:
        image_paths = image_paths[: args.max_samples]
        depth_paths = depth_paths[: args.max_samples]
        print(f"[INFO] Limiting to {args.max_samples} samples")

    _, _, test_loader, _, _ = get_dataloader(
        image_paths, depth_paths, batch_size, transform, dataset_type=1
    )

    loss_fn = nn.CrossEntropyLoss()

    targets = EXPERIMENTS
    if args.group == "default":
        targets = [e for e in targets if e["sched"]]
    elif args.group == "adjusted":
        targets = [e for e in targets if not e["sched"]]
    if args.only:
        targets = [e for e in targets if exp_dir_name(e) == args.only]

    print(f"[INFO] {len(targets)} experiment(s) to run  |  device={device}")

    for exp_i, cfg in enumerate(targets, 1):
        name = exp_dir_name(cfg)
        exp_path = os.path.join("experiments", name)

        if not os.path.isdir(exp_path):
            print(f"[SKIP] {name}  (dir not found)")
            continue
        if not os.path.isfile(os.path.join(exp_path, cfg["ckpt"])):
            print(f"[SKIP] {name}  (checkpoint {cfg['ckpt']} not found)")
            continue

        log_path = os.path.join(exp_path, "inference.log")
        logger = make_logger(log_path)

        lr_tag = "default (lr=0.001)" if cfg["sched"] else "adjusted (0.01→0.001)"
        logger.info(f"Experiment       : {name}")
        logger.info(f"LR schedule      : {lr_tag}")
        logger.info(f"Checkpoint       : {cfg['ckpt']}")
        logger.info(f"Learnable α/β    : {cfg['learn']}")
        logger.info(f"Device           : {device}")
        logger.info(f"Batch size       : {batch_size}")
        logger.info(f"Seed             : {args.seed}")
        logger.info(f"Test samples     : {len(test_loader.dataset)}")
        logger.info("---")

        print(f"\n[{exp_i}/{len(targets)}] {name}")
        print(f"  checkpoint: {cfg['ckpt']}  |  lr: {lr_tag}")

        override = {"learnable_alpha_beta": True} if cfg["learn"] else None
        try:
            _, model, _, _, _, _ = load_experiment(
                name, checkpoint_name=cfg["ckpt"],
                depth=True, map_location=device, override_config=override,
            )
        except Exception as e:
            print(f"LOAD ERROR: {e}")
            logger.info(f"[ERROR] model load: {e}")
            for h in logger.handlers:
                h.close()
            continue

        model = model.to(device)
        acc, avg_loss, total_time, attn_data = timed_inference(
            model, test_loader, loss_fn, device, logger, exp_label=name
        )

        # CMI
        cmi_path = os.path.join("vclub_model", f"{cfg['base']}_CMI_dim_65.pth")
        if os.path.isfile(cmi_path):
            try:
                ck = torch.load(cmi_path, map_location=device, weights_only=True)
                cc = ck["config"]
                cmi_model = ConditionalCLUB(
                    cc["x_dim"], cc["y_dim"], cc["num_classes"], cc["hidden_size"]
                ).to(device)
                cmi_model.load_state_dict(ck["model_state_dict"])
                cmi_model.eval()
                f_r, f_d, cmi_labels = process_attention_data_forCMI(attn_data, device)
                cmi_val = cmi_model.calculate_cmi(f_r, f_d, cmi_labels)
                logger.info(f"CMI              : {cmi_val:.4f}")
                del cmi_model
            except Exception as e:
                logger.info(f"CMI failed: {e}")

        print(f"  => acc={acc:.4f}  time={total_time:.1f}s")

        for h in logger.handlers:
            h.close()
        del model, attn_data
        gc.collect()
        torch.cuda.empty_cache()

    print("[DONE] All experiments finished.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inference cost logger")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit test set size (e.g. 200 for quick check)")
    p.add_argument("--group", choices=["all", "default", "adjusted"], default="all",
                   help="Run only default-lr / adjusted-lr / all experiments")
    p.add_argument("--only", type=str, default=None,
                   help="Run a single experiment by dir name (e.g. NYU_latefusion_lr1e-3)")
    run(p.parse_args())
