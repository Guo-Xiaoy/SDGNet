import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
import torch
from easydict import EasyDict as edict

# 你项目里的模块
from data import get_dataset, get_dataloader
from models import NgeNet, architectures
from utils import decode_config


# ============================================================
# Utils
# ============================================================
def move_to_device(batch, device):
    """Recursively move dict(list/tensor) to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, list):
            new_list = []
            for item in v:
                if torch.is_tensor(item):
                    new_list.append(item.to(device, non_blocking=True))
                else:
                    new_list.append(item)
            out[k] = new_list
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def summarize_inputs(inputs):
    """Print a concise summary of dynamic shapes that affect FLOPs/latency."""
    def shp(x):
        return tuple(x.shape) if torch.is_tensor(x) else str(type(x))

    print("    -> Input summary (dynamic shapes):")
    for key in ["points", "neighbors", "pools", "upsamples", "stacked_lengths", "feats", "normals"]:
        if key not in inputs:
            continue
        v = inputs[key]
        if isinstance(v, list):
            shapes = []
            for i, t in enumerate(v):
                if torch.is_tensor(t):
                    shapes.append(f"L{i}:{tuple(t.shape)}")
                else:
                    shapes.append(f"L{i}:{type(t)}")
            print(f"       - {key}: " + ", ".join(shapes))
        else:
            print(f"       - {key}: {shp(v)}")


# ============================================================
# FLOPs computation methods
# ============================================================
def compute_flops_fvcore(model, inputs):
    """fvcore FLOPs (may fail on dynamic shapes)."""
    from fvcore.nn import FlopCountAnalysis
    flops_analyzer = FlopCountAnalysis(model, (inputs,))
    flops_analyzer.unsupported_ops_warnings(False)
    flops = flops_analyzer.total()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return flops, params


def compute_flops_thop(model, inputs):
    """
    thop FLOPs proxy. Robustly initializes total_ops for all modules and
    uses Python int accumulation to avoid version mismatches across thop versions.
    """
    from thop import profile
    import torch.nn as nn

    # 1) Force-init total_ops for every module (thop may skip it in some versions)
    for m in model.modules():
        setattr(m, "total_ops", 0)

    def count_linear(m, x, y):
        x = x[0]
        # MACs for linear: batch * in_features * out_features
        # If x is (B, *, in_features), treat all leading dims as "batch-like"
        if x.dim() >= 2:
            batch_like = int(x.numel() // x.shape[-1])
        else:
            batch_like = int(x.shape[0]) if x.numel() > 0 else 1
        macs = batch_like * m.in_features * m.out_features
        m.total_ops += int(macs)

    custom_ops = {nn.Linear: count_linear}

    macs, params = profile(model, inputs=(inputs,), custom_ops=custom_ops, verbose=False)

    # thop returns MACs for many ops; convert to FLOPs by *2 (mul+add)
    flops = float(macs) * 2.0
    return flops, int(params)


def compute_flops_torch_flopcounter(model, inputs):
    """
    PyTorch runtime FLOPs counter. Avoid inference_mode to prevent
    'Cannot set version_counter for inference tensor' errors.
    """
    from torch.utils.flop_counter import FlopCounterMode

    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        with FlopCounterMode(display=False) as fcm:
            _ = model(inputs)

    total_flops = fcm.get_total_flops()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if model_was_training:
        model.train()
    return int(total_flops), int(params)

def compute_flops_profiler(model, inputs, device):
    """
    Runtime estimate via torch.profiler. Might miss custom CUDA ops,
    but often runs when others fail.
    """
    import torch.profiler as prof

    model.eval()
    with torch.no_grad():
        with prof.profile(
            activities=[prof.ProfilerActivity.CPU] + ([prof.ProfilerActivity.CUDA] if device.type=="cuda" else []),
            with_flops=True,
            record_shapes=True
        ) as p:
            _ = model(inputs)

    # Not all ops report flops; sum what's available
    total_flops = 0
    for evt in p.key_averages():
        if hasattr(evt, "flops") and evt.flops is not None:
            total_flops += evt.flops
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total_flops), int(params)


# ============================================================
# Latency measurement
# ============================================================
def measure_latency_ms(model, inputs, device, warmup=10, runs=50):
    model.eval()
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            for _ in range(warmup):
                _ = model(inputs)
            torch.cuda.synchronize()

            times = []
            for _ in range(runs):
                starter.record()
                _ = model(inputs)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))  # ms
        return float(np.mean(times)), float(np.std(times))
    else:
        # CPU fallback
        with torch.inference_mode():
            for _ in range(warmup):
                _ = model(inputs)

            t0 = time.time()
            for _ in range(runs):
                _ = model(inputs)
            t1 = time.time()
        avg = (t1 - t0) / runs * 1000.0
        return float(avg), 0.0


# ============================================================
# Main
# ============================================================
def main():
    # ---- config ----
    config_path = "./configs/customdata.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    print(f"Loading config from {config_path}...")
    config = decode_config(config_path)
    config = edict(config)
    config.architecture = architectures[config.dataset]

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- model ----
    print("Building model...")
    model = NgeNet(config).to(device)
    model.eval()

    # ---- data (one batch) ----
    print("Loading data...")
    # 重要：固定 neighborhood_limits，减少动态变化，便于 FLOPs 统计稳定
    neighborhood_limits = [8, 15, 22, 23]

    _, _, test_dataset = get_dataset(config.dataset, config)
    test_loader, _ = get_dataloader(
        config=config,
        dataset=test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        neighborhood_limits=neighborhood_limits,  # <-- 原代码这里是 None，现在改成固定值
    )

    batch = next(iter(test_loader))
    inputs = move_to_device(batch, device)

    print("\n" + "=" * 50)
    print("  SDGNet Efficiency Benchmark  ")
    print("=" * 50)

    summarize_inputs(inputs)

    # ---- Params ----
    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[1] Parameters: {params_total / 1e6:.2f} M")

    # ---- FLOPs ----
    print("[2] Measuring FLOPs...")
    flops_val = None
    flops_method = None

    # A) fvcore
    try:
        print("    -> Trying fvcore...")
        flops_val, _ = compute_flops_fvcore(model, inputs)
        flops_method = "fvcore"
        print(f"    -> Success ({flops_method})! FLOPs: {flops_val / 1e9:.3f} G")
    except Exception as e:
        print(f"    -> fvcore failed: {str(e)[:160]}...")

    # B) thop
    if flops_val is None:
        try:
            print("    -> Trying thop (custom Linear handler)...")
            flops_val, _ = compute_flops_thop(model, inputs)
            flops_method = "thop"
            print(f"    -> Success ({flops_method})! FLOPs: {flops_val / 1e9:.3f} G")
        except Exception as e:
            print(f"    -> thop failed: {str(e)[:160]}...")

    # C) PyTorch flop_counter (recommended fallback)
    if flops_val is None:
        try:
            print("    -> Trying torch.utils.flop_counter (runtime estimate)...")
            flops_val, _ = compute_flops_torch_flopcounter(model, inputs)
            flops_method = "torch_flop_counter"
            print(f"    -> Success ({flops_method})! FLOPs: {flops_val / 1e9:.3f} G")
            print("    -> Note: some custom CUDA / indexing ops may be uncounted (estimate).")
        except Exception as e:
            print(f"    -> torch flop_counter failed: {str(e)[:160]}...")
            print("    -> [Fallback] FLOPs may be unavailable due to dynamic point-cloud ops.")
            print("       Consider reporting Params + Latency, and specify the input setting (N,k).")

    # ---- Latency ----
    print("[3] Measuring Latency (Batch Size = 1)...")
    avg_ms, std_ms = measure_latency_ms(model, inputs, device, warmup=10, runs=50)
    if std_ms > 0:
        print(f"    -> Average Latency: {avg_ms:.2f} ms (± {std_ms:.2f} ms)")
    else:
        print(f"    -> Average Latency: {avg_ms:.2f} ms")

    print("=" * 50)
    if flops_val is not None:
        print(f"Summary: Params={params_total/1e6:.2f}M, FLOPs={flops_val/1e9:.3f}G ({flops_method}), Latency={avg_ms:.2f}ms")
    else:
        print(f"Summary: Params={params_total/1e6:.2f}M, FLOPs=NA, Latency={avg_ms:.2f}ms")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
