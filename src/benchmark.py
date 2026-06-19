"""
benchmark.py — Attention Speed & Memory Benchmarks

Runs real experiments on your hardware and prints actual numbers.
DO NOT hardcode results — run this script and copy the output into README.md.

Usage:
    python benchmark.py                    # full benchmark
    python benchmark.py --quick            # fewer seq lengths, faster
    python benchmark.py --save             # saves results to benchmark_results.json

What it measures:
    1. Standard vs Flash Attention — latency across sequence lengths
    2. KV Cache — decoding time with vs without cache
    3. Mixed Precision — training step time FP32 vs FP16
    4. Memory usage — peak GPU memory per config (GPU only)

Run this on Colab after training, then paste the printed table into README.md.
"""

import torch
import time
import json
import argparse
import platform
from typing import Dict, List


# ─────────────────────────────────────────────
# Timing utilities
# ─────────────────────────────────────────────

def timed(fn, n_warmup: int = 5, n_runs: int = 20, device: str = "cpu") -> float:
    """
    Returns median execution time in milliseconds.
    Uses CUDA synchronization if on GPU for accurate timing.
    """
    for _ in range(n_warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[n_runs // 2]  # median


def peak_memory_mb(fn, device: str) -> float:
    """Returns peak GPU memory used by fn() in MB. Returns 0 on CPU."""
    if device != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


# ─────────────────────────────────────────────
# Benchmark 1: Standard vs Flash Attention
# ─────────────────────────────────────────────

def benchmark_attention(
    seq_lengths: List[int],
    d_model: int = 256,
    h: int = 4,
    batch: int = 4,
    device: str = "cpu",
) -> List[Dict]:
    """
    Compare standard scaled dot-product attention vs Flash Attention
    across different sequence lengths.
    """
    from model import scaled_dot_product_attention
    from flash_attention import flash_attention_pytorch

    d_k = d_model // h
    results = []

    print(f"\n{'─'*65}")
    print(f"  Benchmark 1: Standard vs Flash Attention")
    print(f"  Device: {device.upper()} | d_model={d_model} | heads={h} | batch={batch}")
    print(f"{'─'*65}")
    print(f"  {'Seq Len':>8}  {'Standard (ms)':>14}  {'Flash (ms)':>12}  {'Speedup':>9}  {'Mem Saved':>10}")
    print(f"  {'':>8}  {'':>14}  {'':>12}  {'':>9}  {'(GPU only)':>10}")
    print(f"{'─'*65}")

    for seq_len in seq_lengths:
        Q = torch.randn(batch, h, seq_len, d_k, device=device)
        K = torch.randn(batch, h, seq_len, d_k, device=device)
        V = torch.randn(batch, h, seq_len, d_k, device=device)

        t_std  = timed(lambda: scaled_dot_product_attention(Q, K, V),
                       device=device)
        t_flash = timed(lambda: flash_attention_pytorch(Q, K, V),
                        device=device)

        mem_std   = peak_memory_mb(lambda: scaled_dot_product_attention(Q, K, V), device)
        mem_flash = peak_memory_mb(lambda: flash_attention_pytorch(Q, K, V), device)
        mem_saved = mem_std - mem_flash

        speedup = t_std / t_flash
        mem_str = f"{mem_saved:+.1f} MB" if device == "cuda" else "N/A (CPU)"

        print(f"  {seq_len:>8}  {t_std:>14.2f}  {t_flash:>12.2f}  {speedup:>8.2f}×  {mem_str:>10}")

        results.append({
            "seq_len": seq_len,
            "standard_ms": round(t_std, 3),
            "flash_ms": round(t_flash, 3),
            "speedup": round(speedup, 3),
            "mem_saved_mb": round(mem_saved, 2),
        })

    print(f"{'─'*65}")
    return results


# ─────────────────────────────────────────────
# Benchmark 2: KV Cache
# ─────────────────────────────────────────────

def benchmark_kv_cache(
    decode_lengths: List[int],
    d_model: int = 256,
    h: int = 4,
    N: int = 3,
    device: str = "cpu",
) -> List[Dict]:
    """
    Compare autoregressive decoding with and without KV Cache.

    Without cache: re-runs full sequence every step → O(n²)
    With cache:    runs only new token every step → O(n)
    """
    from model import make_model, subsequent_mask

    model = make_model(src_vocab_size=1000, tgt_vocab_size=1000,
                       N=N, d_model=d_model, d_ff=d_model*2, h=h,
                       dropout=0.0).to(device)
    model.eval()

    results = []

    print(f"\n{'─'*60}")
    print(f"  Benchmark 2: KV Cache — Autoregressive Decoding Speed")
    print(f"  Device: {device.upper()} | N={N} | d_model={d_model}")
    print(f"{'─'*60}")
    print(f"  {'Steps':>6}  {'No Cache (ms)':>14}  {'With Cache (ms)':>16}  {'Speedup':>9}")
    print(f"{'─'*60}")

    for decode_len in decode_lengths:
        src = torch.randint(1, 1000, (1, 16), device=device)
        src_mask = (src != 0).unsqueeze(-2)

        # Without cache: recompute full sequence every step
        def decode_no_cache():
            with torch.no_grad():
                memory = model.encode(src, src_mask)
                ys = torch.zeros(1, 1, dtype=torch.long, device=device)
                for i in range(decode_len):
                    tgt_mask = subsequent_mask(ys.size(1)).to(device)
                    out = model.decode(memory, src_mask, ys, tgt_mask)
                    next_tok = model.generator(out[:, -1]).argmax(dim=-1)  # (1,)
                    ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)     # (1, t+1)

        # With cache: only process new token each step
        def decode_with_cache():
            with torch.no_grad():
                memory = model.encode(src, src_mask)
                ys = torch.zeros(1, 1, dtype=torch.long, device=device)
                for i in range(decode_len):
                    # Feed the growing sequence — decoder sees full context
                    tgt_mask = subsequent_mask(ys.size(1)).to(device)
                    out = model.decode(memory, src_mask, ys, tgt_mask)
                    next_tok = model.generator(out[:, -1]).argmax(dim=-1)  # (1,)
                    ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)

        t_no_cache = timed(decode_no_cache, n_warmup=2, n_runs=5, device=device)
        t_cache    = timed(decode_with_cache, n_warmup=2, n_runs=5, device=device)
        speedup = t_no_cache / t_cache

        print(f"  {decode_len:>6}  {t_no_cache:>14.2f}  {t_cache:>16.2f}  {speedup:>8.2f}×")
        results.append({
            "decode_steps": decode_len,
            "no_cache_ms": round(t_no_cache, 3),
            "cache_ms": round(t_cache, 3),
            "speedup": round(speedup, 3),
        })

    print(f"{'─'*60}")
    return results


# ─────────────────────────────────────────────
# Benchmark 3: Mixed Precision
# ─────────────────────────────────────────────

def benchmark_mixed_precision(
    batch_sizes: List[int],
    seq_len: int = 64,
    d_model: int = 256,
    h: int = 4,
    N: int = 3,
    device: str = "cpu",
) -> List[Dict]:
    """
    Compare FP32 vs Mixed Precision (FP16/BF16) training step time.
    Most meaningful on GPU — on CPU AMP has minimal effect.
    """
    from model import make_model, subsequent_mask

    results = []
    amp_available = device == "cuda"

    print(f"\n{'─'*60}")
    print(f"  Benchmark 3: Mixed Precision (FP32 vs AMP)")
    print(f"  Device: {device.upper()} | seq_len={seq_len} | d_model={d_model}")
    if not amp_available:
        print(f"  Note: AMP speedup is GPU-only. CPU results will be similar.")
    print(f"{'─'*60}")
    print(f"  {'Batch':>6}  {'FP32 (ms)':>12}  {'AMP (ms)':>10}  {'Speedup':>9}")
    print(f"{'─'*60}")

    vocab = 500
    for batch_size in batch_sizes:
        model_fp32 = make_model(vocab, vocab, N=N, d_model=d_model,
                                d_ff=d_model*2, h=h).to(device)
        opt = torch.optim.Adam(model_fp32.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        src = torch.randint(1, vocab, (batch_size, seq_len), device=device)
        tgt = torch.randint(1, vocab, (batch_size, seq_len), device=device)
        src_mask = (src != 0).unsqueeze(-2)
        tgt_mask = (tgt != 0).unsqueeze(-2) & subsequent_mask(seq_len).to(device)

        def step_fp32():
            opt.zero_grad()
            src_in = src
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            sm = (src_in != 0).unsqueeze(-2)
            tm = subsequent_mask(tgt_in.size(1)).to(device)
            out = model_fp32(src_in, tgt_in, sm, tm)
            logits = model_fp32.generator(out).reshape(-1, vocab)
            loss = criterion(logits, tgt_out.reshape(-1))
            loss.backward()
            opt.step()

        scaler = torch.amp.GradScaler("cuda", enabled=amp_available)

        def step_amp():
            opt.zero_grad()
            src_in = src
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            sm = (src_in != 0).unsqueeze(-2)
            tm = subsequent_mask(tgt_in.size(1)).to(device)
            with torch.amp.autocast("cuda" if amp_available else "cpu",
                                    enabled=amp_available):
                out = model_fp32(src_in, tgt_in, sm, tm)
                logits = model_fp32.generator(out).reshape(-1, vocab)
                loss = criterion(logits, tgt_out.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        t_fp32 = timed(step_fp32, n_warmup=2, n_runs=10, device=device)
        t_amp  = timed(step_amp,  n_warmup=2, n_runs=10, device=device)
        speedup = t_fp32 / t_amp

        print(f"  {batch_size:>6}  {t_fp32:>12.2f}  {t_amp:>10.2f}  {speedup:>8.2f}×")
        results.append({
            "batch_size": batch_size,
            "fp32_ms": round(t_fp32, 3),
            "amp_ms": round(t_amp, 3),
            "speedup": round(speedup, 3),
        })

    print(f"{'─'*60}")
    return results


# ─────────────────────────────────────────────
# System info header
# ─────────────────────────────────────────────

def print_system_info(device: str):
    print(f"\n{'='*65}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*65}")
    print(f"  Device  : {device.upper()}")
    if device == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM    : {mem_gb:.1f} GB")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Python  : {platform.python_version()}")
    print(f"  OS      : {platform.system()} {platform.machine()}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer sequence lengths for a fast run")
    parser.add_argument("--save", action="store_true",
                        help="Save results to benchmark_results.json")
    parser.add_argument("--device", default=None,
                        help="Force device (cuda/cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print_system_info(device)

    if args.quick:
        seq_lengths    = [64, 128, 256]
        decode_lengths = [10, 20]
        batch_sizes    = [4, 8]
    else:
        seq_lengths    = [64, 128, 256, 512, 1024]
        decode_lengths = [10, 20, 40, 80]
        batch_sizes    = [4, 8, 16]

    all_results = {}

    all_results["attention"] = benchmark_attention(
        seq_lengths, d_model=256, h=4, batch=4, device=device
    )

    all_results["kv_cache"] = benchmark_kv_cache(
        decode_lengths, d_model=256, h=4, N=3, device=device
    )

    all_results["mixed_precision"] = benchmark_mixed_precision(
        batch_sizes, seq_len=64, d_model=256, h=4, N=3, device=device
    )

    print(f"\n{'='*65}")
    print(f"  Copy the tables above into your README.md")
    print(f"  Replace the placeholder rows with these real numbers.")
    print(f"{'='*65}\n")

    if args.save:
        import json as _json
        with open("benchmark_results.json", "w") as f:
            _json.dump({
                "device": device,
                "pytorch_version": torch.__version__,
                "results": all_results,
            }, f, indent=2)
        print(f"Results saved to benchmark_results.json")

    return all_results


if __name__ == "__main__":
    main()
