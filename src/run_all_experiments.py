"""
run_all_experiments.py — One-Shot Colab Runner

Runs the full Tier-1 pipeline in sequence so you can fill in EXPERIMENTS.md
in one Colab session instead of running scripts one at a time:

    1. Train the baseline model           → train.py
    2. Evaluate BLEU (greedy + beam)       → evaluate.py
    3. Run latency/memory benchmarks       → benchmark.py
    4. Generate attention heatmaps         → visualize.py

Usage (Colab):
    !python run_all_experiments.py

Everything prints to console — copy the relevant blocks straight into
EXPERIMENTS.md. Nothing here is invented; if a step fails, it stops and
tells you exactly what to fix rather than silently skipping.
"""

import os
import sys
import time
import torch


SEPARATOR = "=" * 70


def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. Training will be slow.")
        print("In Colab: Runtime → Change runtime type → T4 GPU")

    total_start = time.time()

    # ── Step 1: Train ────────────────────────────────────────────
    section("STEP 1 / 5 — Training baseline model")
    print("This trains N=3, d_model=256 on Multi30k for 8 epochs.")
    print("Expect ~15-25 minutes on a free Colab T4.\n")

    from train import train as run_training
    model, vocab_src, vocab_tgt = run_training()

    # ── Step 2: Evaluate BLEU ────────────────────────────────────
    section("STEP 2 / 5 — Evaluating BLEU (greedy + beam search)")

    from train import create_dataloaders, load_tokenizers, CONFIG
    from evaluate import evaluate_bleu

    spacy_de, spacy_en = load_tokenizers()
    _, val_loader = create_dataloaders(
        spacy_de, spacy_en, vocab_src, vocab_tgt, CONFIG, device
    )

    print("\n--- Greedy decoding BLEU ---")
    result_greedy = evaluate_bleu(
        model, val_loader, vocab_src, vocab_tgt, device,
        use_beam=False, n_examples=200,
    )

    print("\n--- Beam search (k=4) BLEU ---")
    result_beam = evaluate_bleu(
        model, val_loader, vocab_src, vocab_tgt, device,
        use_beam=True, beam_size=4, n_examples=200,
    )

    print(f"\n>>> COPY THIS INTO EXPERIMENTS.md <<<")
    print(f"BLEU (greedy): {result_greedy['bleu']:.2f}")
    print(f"BLEU (beam k=4): {result_beam['bleu']:.2f}")
    print(f"Delta from beam search: {result_beam['bleu'] - result_greedy['bleu']:+.2f}")

    # ── Step 3: Benchmark ────────────────────────────────────────
    section("STEP 3 / 5 — Latency & memory benchmarks")

    import benchmark
    print_system_info_fn = benchmark.print_system_info
    print_system_info_fn(device)

    seq_lengths = [128, 256, 512, 1024] if device == "cuda" else [64, 128, 256]
    decode_lengths = [20, 40, 80]
    batch_sizes = [8, 16] if device == "cuda" else [4, 8]

    attn_results = benchmark.benchmark_attention(
        seq_lengths, d_model=256, h=4, batch=4, device=device
    )
    kv_results = benchmark.benchmark_kv_cache(
        decode_lengths, d_model=256, h=4, N=3, device=device
    )
    amp_results = benchmark.benchmark_mixed_precision(
        batch_sizes, seq_len=64, d_model=256, h=4, N=3, device=device
    )

    import json
    with open("benchmark_results.json", "w") as f:
        json.dump({
            "device": device,
            "pytorch_version": torch.__version__,
            "results": {
                "attention": attn_results,
                "kv_cache": kv_results,
                "mixed_precision": amp_results,
            },
        }, f, indent=2)
    print("\nSaved: benchmark_results.json")

    # ── Step 4: Visualize attention ──────────────────────────────
    section("STEP 4 / 5 — Generating real attention heatmaps")

    from visualize import visualize_translation

    demo_sentences = [
        "Ein Hund läuft durch den Park.",
        "Zwei Männer spielen Fußball.",
    ]

    for sent in demo_sentences:
        try:
            visualize_translation(
                sent, model, vocab_src, vocab_tgt, spacy_de, device,
                save_dir="attention_maps",
            )
        except Exception as e:
            print(f"Visualization failed for '{sent}': {e}")
            print("(Non-fatal — training/eval/benchmark results above are still valid)")

    # ── Step 5: Profiler ─────────────────────────────────────────
    section("STEP 5 / 5 — Per-layer profiling")

    try:
        from profile_model import run_profiler
        run_profiler(seq_len=128, batch_size=8, N=CONFIG["N"],
                    d_model=CONFIG["d_model"], d_ff=CONFIG["d_ff"],
                    h=CONFIG["h"], n_steps=10, device=device)
    except Exception as e:
        print(f"Profiling failed: {e}")
        print("(Non-fatal — run `python profile_model.py` separately if needed)")

    # ── Done ──────────────────────────────────────────────────────
    total_time = time.time() - total_start
    section(f"ALL DONE in {total_time/60:.1f} minutes")

    print("""
Next steps:
  1. Open EXPERIMENTS.md and fill in Experiment 1 with the BLEU numbers above
  2. Open RESULTS.md and fill in the publication-style summary
  3. Open README.md and replace the placeholder benchmark tables with the
     real numbers printed in Step 3 (also saved in benchmark_results.json)
  4. Replace assets/attention_sample.png with a real heatmap from
     attention_maps/ (generated in Step 4)
  5. Check profiler_summary.txt and profiler_trace.json (Step 5) — open
     the trace at https://ui.perfetto.dev for a visual timeline
  6. Commit everything: checkpoints excluded (.gitignore), but assets/,
     EXPERIMENTS.md, RESULTS.md, and benchmark_results.json should all
     be pushed
""")


if __name__ == "__main__":
    main()
