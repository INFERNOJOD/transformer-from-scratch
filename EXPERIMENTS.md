# Experiments Log

This file is a running record of actual training runs. Every entry below was
produced by running `train.py` / `evaluate.py` / `benchmark.py` on real hardware
and pasting the real output — no invented numbers.

**Workflow for each entry:**
1. Run `python train.py` with the config noted below
2. Run `python evaluate.py --checkpoint <path>` for BLEU
3. Paste the actual printed output into this file
4. Note what changed vs the previous experiment and why

---

## Experiment 0 — Sanity Check

**Date:** _(fill in)_
**Goal:** Confirm the full pipeline runs end-to-end without errors.

```
Config: N=2, d_model=64, d_ff=128, h=4, epochs=1, subset of Multi30k
```

**Result:**
```
(paste console output here — even a 1-epoch smoke run on a few hundred
sentences is enough to confirm: data loads, forward pass works, loss
decreases, checkpoint saves, inference produces output)
```

**Notes:** _(anything that broke and how you fixed it — this is useful
material for "tell me about a bug you ran into" interview questions)_

---

## Experiment 1 — Baseline

**Date:** _(fill in actual dates)_
**Config:** `ModelConfig.small()` — N=3, d_model=256, d_ff=512, h=4
**Training:** 8 epochs, batch_size=128, warmup=400, Multi30k De→En

**Command:**
```bash
python train.py
```

**Result — Run A:**
```
Final Train Loss: 1.2656
Final Val Loss:    1.5462
Parameters:        9,358,576
Training data:     29,000 train / 1,014 val / 1,000 test
```

**Result — Run B (independent retrain, same config, verifies reproducibility):**
```
Epoch 1/8 → Train loss: 6.26   | Val loss: 4.57
Epoch 2/8 → Train loss: 3.87   | Val loss: 3.24
Epoch 3/8 → Train loss: 2.8306 | Val loss: 2.3784
Epoch 4/8 → Train loss: 2.1605 | Val loss: 1.9279
Epoch 5/8 → Train loss: 1.7686 | Val loss: 1.7071
Epoch 6/8 → Train loss: 1.5349 | Val loss: 1.6080
Epoch 7/8 → Train loss: 1.3841 | Val loss: 1.5720
Epoch 8/8 → Train loss: 1.2729 | Val loss: 1.5295
```

**Result — Run C (fresh clone verification, used for final BLEU):**
```
Final Val Loss: 1.5295  (matches Run B exactly — same checkpoint reused)
BLEU (greedy):  not separately measured (only beam search was run — see Exp 2)
```

**Sample translations (Run B/C checkpoint, greedy):**
```
DE: Ein Hund läuft durch den Park.
EN: A dog is running through the park.

DE: Zwei Männer spielen Fußball.
EN: Two men are playing soccer.

DE: Ein kleines Kind sitzt auf einer Schaukel.
EN: A young child is sitting on a swing.

DE: Eine Frau liest ein Buch im Café.
EN: A woman reading a book at a cafe reading a book.   ← repetition artifact

DE: Der Zug fährt durch den Tunnel.
EN: The train is going through the tunnel.
```

**Notes:** Three separate runs total. Run A and Run B/C are independent
full trainings from scratch — final val loss differs by only ~1.1%
(1.5462 vs 1.5295), confirming the pipeline converges reproducibly rather
than depending on a lucky seed. Run C reused Run B's checkpoint
(`epoch_08.pt`) for a from-scratch-clone verification of the full
pipeline (install → test → train-skip-if-checkpoint-exists → inference →
evaluate → visualize → benchmark → profile) — see Experiment 2 for that
verification's BLEU result. One real artifact worth noting: the 4th demo
sentence shows the model repeating "reading a book" — a genuine,
reproducible quirk worth understanding, not edited out.

---

## Experiment 2 — + Beam Search

**Date:** _(fill in actual dates)_
**Change:** Same trained checkpoint as Experiment 1, decode with beam search
(k=4, length penalty α=0.6) instead of greedy.

**Command:**
```bash
python evaluate.py --checkpoint checkpoints/epoch_08.pt --beam 4
```

**Result — Run 1:**
```
BLEU (beam, k=4): 36.30
Breakdown:        69.3/45.6/30.5/21.0 (1-gram/2-gram/3-gram/4-gram precision)
Brevity penalty:  0.963 (ratio=0.964, hyp_len=13261, ref_len=13762)
Evaluated on:     full validation set (1,014 examples)
Eval time:        ~8m37s (1.96 it/s on Colab T4)
```

**Result — Run 2 (fresh clone, independent re-verification):**
```
BLEU (beam, k=4): 36.37
Breakdown:        66.9/44.2/29.6/20.0
Brevity penalty:  1.000 (ratio=1.013, hyp_len=13937, ref_len=13762)
Evaluated on:     full validation set (1,014 examples)
Eval time:        ~8m36s (1.96 it/s on Colab T4)
```

**BLEU (greedy):** not yet measured separately. To isolate beam search's
exact contribution, run:
```bash
python evaluate.py --checkpoint checkpoints/epoch_08.pt --beam 0
```
and record the result here. Until that's done, the delta from beam search
specifically (vs greedy) is an open item — don't guess at it.

**Notes:** Two independent end-to-end evaluations (different checkpoints
from Experiment 1's two training runs) produced near-identical BLEU
(36.30 vs 36.37, within 0.2%) — strong evidence the evaluation pipeline
itself is stable and not the source of variance between runs. Interesting
secondary detail: brevity penalty differs (0.963 vs 1.000) — Run 2's
model produced slightly longer outputs relative to references
(hyp_len=13937 vs 13261), suggesting marginally different length
calibration between the two training runs despite nearly identical BLEU.

**Bug found and fixed along the way:** an earlier version of `evaluate.py`
was library-only (no CLI entry point) — running
`python evaluate.py --checkpoint ...` printed nothing because there was no
`main()` / `argparse` / `if __name__ == "__main__":` block. This was
caught when a Colab cell showed 2 seconds of runtime with zero output
(impossible for a real 1,014-example beam-search BLEU run, which takes
8+ minutes). Fixed by adding a proper CLI wrapper; verified working via
`--help` and a full successful run (Run 2 above) immediately after.
Worth remembering as a real "bug I found and fixed" story for interviews —
the diagnostic signal was the runtime/output mismatch, not an error
message.

---

## Planned Experiment 3 — + RoPE

**Date:** _(fill in)_
**Change:** Replace sinusoidal PE with RoPE. Retrain from scratch,
same config otherwise (N=3, d_model=256, 8 epochs).

**Command:**
```python
# in train.py, swap MultiHeadAttention for RoPEMultiHeadAttention
```

**Result:**
```
Val Loss:  ___  (baseline was ___)
BLEU:      ___  (baseline was ___)
```

**Notes:** _(Did it converge faster/slower than baseline? Any difference
in training stability? This is what you'd discuss if asked "did RoPE
actually help here")_

---

## Planned Experiment 4 — + RMSNorm + SwiGLU

**Date:** _(fill in)_
**Change:** Swap LayerNorm → RMSNorm, ReLU FFN → SwiGLU FFN.

**Result:**
```
Val Loss:  ___
BLEU:      ___
Params:    ___  (SwiGLU uses 3 matrices vs FFN's 2 — note the difference)
```

---

## Planned Experiment 5 — + Weight Tying

**Date:** _(fill in)_
**Change:** Apply `apply_weight_tying()` after model creation.

**Result:**
```
Params before tying: ___
Params after tying:  ___
Val Loss:  ___
BLEU:      ___
```

---

## Experiment 6 — Flash Attention Speed (no retraining needed)

**Date:** _(fill in actual dates)_
**Goal:** Verify Flash Attention gives identical outputs but faster inference.

**Command:**
```bash
python benchmark.py --save
```

**Result — Run 1:**
```
Seq Length | Standard (ms) | Flash (ms) | Speedup | Mem Saved
-----------|---------------|------------|---------|----------
64         | 0.106         | 0.049      | 2.15×   | 0.5 MB
128        | 0.123         | 0.077      | 1.61×   | 2.0 MB
256        | 0.264         | 0.185      | 1.43×   | 8.0 MB
512        | 0.709         | 0.653      | 1.09×   | 32.0 MB
1024       | 2.652         | 2.224      | 1.19×   | 128.0 MB

GPU: Tesla T4
PyTorch: 2.11.0+cu128
```

**Result — Run 2 (independent re-run):**
```
Seq Length | Standard (ms) | Flash (ms) | Speedup | Mem Saved
-----------|---------------|------------|---------|----------
64         | 0.12          | 0.05       | 2.39×   | +0.5 MB
128        | 0.12          | 0.08       | 1.49×   | +2.0 MB
256        | 0.28          | 0.21       | 1.33×   | +8.0 MB
512        | 0.77          | 0.73       | 1.05×   | +32.0 MB
1024       | 2.85          | 2.29       | 1.24×   | +128.0 MB
```

**Notes:** Flash Attention wins on both speed and memory at every sequence
length, in both runs. Speedup isn't perfectly monotonic in either run —
highest at seq=64 rather than at the longest sequence tested — likely
because this model is small (d_model=256, h=4) and at short sequences
PyTorch's per-call overhead exaggerates the relative gap. Memory savings
scale predictably (~4× per doubling of seq_len) in both runs, consistent
with the expected O(seq²) vs O(seq) difference. See `RESULTS.md` for full
analysis.

---

## Experiment 7 — KV Cache Speed (no retraining needed)

**Date:** _(fill in actual dates)_

**Result — Run 1:**
```
Decode Steps | No Cache (ms) | With Cache (ms) | Speedup
-------------|---------------|-----------------|--------
10           | 53.94         | 51.49           | 1.05×
20           | 109.87        | 103.87          | 1.06×
40           | 221.63        | 286.15          | 0.78× (cache SLOWER)
80           | 439.28        | 433.34          | 1.01×
```

**Result — Run 2 (independent re-run):**
```
Steps | No Cache (ms) | With Cache (ms) | Speedup
------|---------------|-----------------|--------
10    | 51.92         | 51.93           | 1.00×
20    | 104.16        | 104.59          | 1.00×
40    | 277.21        | 212.55          | 1.30× (cache FASTER here)
80    | 424.19         | 422.57          | 1.00×
```

**Notes — this is the most important "negative result" in this project,
and worth keeping rather than hiding.** KV cache shows essentially no
speedup at this scale (3 layers, d_model=256, only 10-80 decode steps) in
either run. Critically, the one point that deviates from ~1.00× lands at
*different* step counts in each run (40 steps was slowest in Run 1, but
fastest in Run 2) — this is strong evidence the deviation is GPU
scheduling noise, not a structural property of the cache implementation.
`tests.py`'s `TestKVCache` suite independently confirms the cache produces
numerically correct accumulated tensors, so this is purely a performance
characteristic at small scale, not a correctness issue.

**What I'd try next to get a cleaner signal:** benchmark KV cache at a
larger model size (paper-base: N=6, d_model=512) and longer decode
lengths (200+ tokens), where the O(n²) cost without caching should start
to dominate the fixed per-step overhead and the benefit should become
measurable. This is exactly the kind of follow-up experiment that would
make a strong answer to "what would you try next."

---

## Experiment 8 — Mixed Precision (AMP) Speed (no retraining needed)

**Date:** _(fill in actual dates)_

**Result — Run 1:**
```
Batch Size | FP32 (ms) | AMP (ms) | Speedup
-----------|-----------|----------|--------
4          | 31.46     | 36.95    | 0.85× (AMP SLOWER)
8          | 31.66     | 37.78    | 0.84× (AMP SLOWER)
16         | 32.56     | 38.14    | 0.85× (AMP SLOWER)
```

**Result — Run 2 (independent re-run):**
```
Batch | FP32 (ms) | AMP (ms) | Speedup
------|-----------|----------|--------
4     | 31.92     | 36.44    | 0.88× (AMP SLOWER)
8     | 32.39     | 53.28    | 0.61× (AMP SLOWER)
16    | 40.58     | 60.12    | 0.67× (AMP SLOWER)
```

**Notes — also a real negative result, confirmed in both runs.** AMP was
slower than FP32 at every batch size tested, every time. AMP's speedup
comes from GPU Tensor Cores, which need large matrix dimensions to be
worth their dtype-casting and loss-scaling overhead. At d_model=256 and
batch≤16, the matrices are small enough that this overhead outweighs any
Tensor Core benefit on a T4. Consistent with general guidance that AMP's
advantage scales with model/batch size — expected to look different on
the paper's full base model (d_model=512).

---

## Summary Table

| # | Config | Val Loss | BLEU | Notes |
|---|---|---|---|---|
| 1 | Baseline (greedy) | 1.5295 (Run B/C) / 1.5462 (Run A) | not separately measured | 2 independent training runs, ~1% apart |
| 2 | + Beam Search (k=4) | — | **36.37** (also 36.30) | verified twice end-to-end, incl. fresh clone |
| 3 | + RoPE | — | — | implemented, unit-tested, not yet retrained as ablation |
| 4 | + RMSNorm + SwiGLU | — | — | implemented, unit-tested, not yet retrained as ablation |
| 5 | + Weight Tying | — | — | implemented, unit-tested, not yet isolated as ablation |
| 6 | Flash Attention | — | — | 1.05×–2.39× faster across seq lengths, confirmed in 2 runs |
| 7 | KV Cache | — | — | **no measurable benefit at this scale** — confirmed in 2 runs |
| 8 | Mixed Precision (AMP) | — | — | **slower than FP32 at this scale** — confirmed in 2 runs |

**Verification status:** the full pipeline (install → 39 tests → train →
inference → evaluate → visualize → benchmark → profile) was run
successfully twice, including once from a completely fresh repository
clone — confirming the project is reproducible by someone other than the
original session that built it, not just internally consistent.

---

## What I'd Try Next (if I had more compute/time)

- **Isolate greedy BLEU.** Every BLEU number recorded so far is beam
  search (k=4). Running `evaluate.py --beam 0` would finally let me state
  beam search's exact contribution instead of citing the paper's typical
  "+1-3 BLEU" range as a guess.
- **Re-run KV Cache and AMP benchmarks at paper-base scale** (N=6,
  d_model=512) and with longer decode sequences (200+ tokens). The
  current "no benefit" finding is itself useful, but the natural follow-up
  question is "at what scale does it start to matter" — and I don't have
  that answer yet.
- **Plug RoPE, RMSNorm+SwiGLU, and weight tying into `train.py`** as actual
  retrained ablations, not just unit-tested standalone modules. Right now
  I can prove each component works in isolation, but not what it changes
  about end-to-end translation quality on this specific task.
- **Migrate to a BPE tokenizer.** The `<unk>` failure on "ice fishing hut"
  showed up identically across two independent training runs — a clear,
  reproducible signal that vocabulary coverage (not training variance) is
  the next real bottleneck, and BPE is the standard fix.
