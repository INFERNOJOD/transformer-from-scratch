"""
profile_model.py — torch.profiler Integration

Unlike benchmark.py (which times whole operations end-to-end), this gives a
per-layer breakdown of where time and memory actually go — the kind of
output that looks like real systems engineering, not just a stopwatch.

Usage:
    python profile_model.py
    python profile_model.py --seq-len 512 --batch 8

Outputs:
    - Printed table: top operations by CPU/CUDA time
    - profiler_trace.json — open in chrome://tracing or https://ui.perfetto.dev
    - profiler_summary.txt — saved text summary

WHY THIS MATTERS
------------------
benchmark.py answers: "is Flash Attention faster than standard attention?"
profile_model.py answers: "where exactly does the time go inside one forward
pass — embeddings? attention? FFN? which layer?"

This is the difference between knowing something is slow and knowing why.
"""

import argparse
import torch
import torch.profiler as profiler

from model import make_model, subsequent_mask


def run_profiler(
    seq_len: int = 128,
    batch_size: int = 8,
    N: int = 3,
    d_model: int = 256,
    d_ff: int = 512,
    h: int = 4,
    n_steps: int = 10,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on: {device.upper()}")
    print(f"Config: seq_len={seq_len}, batch={batch_size}, "
          f"N={N}, d_model={d_model}, h={h}\n")

    vocab_size = 1000
    model = make_model(vocab_size, vocab_size, N=N, d_model=d_model,
                       d_ff=d_ff, h=h).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    src = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    tgt = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    src_mask = (src != 0).unsqueeze(-2)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    tgt_mask = (tgt_in != 0).unsqueeze(-2) & subsequent_mask(tgt_in.size(1)).to(device)

    def train_step():
        optimizer.zero_grad()
        out = model(src, tgt_in, src_mask, tgt_mask)
        logits = model.generator(out).reshape(-1, vocab_size)
        loss = criterion(logits, tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss

    # Warmup (profiler should not measure the slow first-call compilation)
    for _ in range(3):
        train_step()
    if device == "cuda":
        torch.cuda.synchronize()

    activities = [profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(profiler.ProfilerActivity.CUDA)

    print("Running profiler...\n")

    with profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(n_steps):
            train_step()
            prof.step()

    # ── Print top operations by time ────────────────────────────
    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"

    print("=" * 90)
    print("  TOP 15 OPERATIONS BY TIME")
    print("=" * 90)
    table = prof.key_averages().table(
        sort_by=sort_key, row_limit=15
    )
    print(table)

    # ── Print top operations by memory ──────────────────────────
    if device == "cuda":
        print("\n" + "=" * 90)
        print("  TOP 15 OPERATIONS BY GPU MEMORY")
        print("=" * 90)
        mem_table = prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=15
        )
        print(mem_table)
    else:
        print("\n" + "=" * 90)
        print("  TOP 15 OPERATIONS BY CPU MEMORY")
        print("=" * 90)
        mem_table = prof.key_averages().table(
            sort_by="self_cpu_memory_usage", row_limit=15
        )
        print(mem_table)

    # ── Save outputs ─────────────────────────────────────────────
    prof.export_chrome_trace("profiler_trace.json")
    print(f"\nChrome trace saved to: profiler_trace.json")
    print(f"  View it at: chrome://tracing  or  https://ui.perfetto.dev")

    with open("profiler_summary.txt", "w") as f:
        f.write(f"Profile config: seq_len={seq_len}, batch={batch_size}, "
               f"N={N}, d_model={d_model}, h={h}, device={device}\n\n")
        f.write("TOP OPERATIONS BY TIME\n")
        f.write(table)
        f.write("\n\n")
        if device == "cuda":
            f.write("TOP OPERATIONS BY GPU MEMORY\n")
        else:
            f.write("TOP OPERATIONS BY CPU MEMORY\n")
        f.write(mem_table)
    print(f"Text summary saved to: profiler_summary.txt")

    return prof


def main():
    parser = argparse.ArgumentParser(description="Profile the Transformer model")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_profiler(
        seq_len=args.seq_len,
        batch_size=args.batch,
        n_steps=args.steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
