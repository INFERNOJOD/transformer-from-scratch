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

**Date:** _(fill in)_
**Config:** `ModelConfig.small()` — N=3, d_model=256, d_ff=512, h=4
**Training:** 8 epochs, batch_size=128, warmup=400, Multi30k De→En

**Command:**
```bash
python train.py
```

**Result:**
```
Train Loss: ___
Val Loss:   ___
BLEU (greedy): ___
Training time: ___ (Colab T4)
```

**Sample translations:**
```
DE: Ein Hund läuft durch den Park.
EN (predicted): ___
EN (reference): A dog runs through the park.
```

**Notes:** This is the reference point every later experiment is compared
against. Don't change anything else when running later experiments —
keep this config fixed so comparisons are fair.

---

## Experiment 2 — + Beam Search

**Date:** _(fill in)_
**Change:** Same trained checkpoint as Experiment 1, decode with beam search
(k=4, length penalty α=0.6) instead of greedy.

**Command:**
```bash
python inference.py --checkpoint checkpoints/epoch_08.pt --beam 4
python evaluate.py --checkpoint checkpoints/epoch_08.pt --beam 4
```

**Result:**
```
BLEU (greedy):       ___  (from Exp 1)
BLEU (beam, k=4):    ___
Delta:               ___
Inference time/sentence: greedy ___ ms vs beam ___ ms
```

**Notes:** Beam search doesn't require retraining — it's a decode-time
change only. The BLEU delta here isolates the effect of search strategy
from the effect of model architecture.

---

## Future Experiment 3 — + RoPE

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

## Future Experiment 4 — + RMSNorm + SwiGLU

**Date:** _(fill in)_
**Change:** Swap LayerNorm → RMSNorm, ReLU FFN → SwiGLU FFN.

**Result:**
```
Val Loss:  ___
BLEU:      ___
Params:    ___  (SwiGLU uses 3 matrices vs FFN's 2 — note the difference)
```

---

## Future Experiment 5 — + Weight Tying

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

**Date:** _(fill in)_
**Goal:** Verify Flash Attention gives identical outputs but faster inference.

**Command:**
```bash
python benchmark.py --save
```

**Result:** _(paste the full table from benchmark.py output)_
```
Seq Length | Standard (ms) | Flash (ms) | Speedup
-----------|---------------|------------|--------
128        | ___           | ___        | ___
256        | ___           | ___        | ___
512        | ___           | ___        | ___
1024       | ___           | ___        | ___

GPU: ___
PyTorch: ___
```

---

## Experiment 7 — KV Cache Speed (no retraining needed)

**Result:** _(paste from benchmark.py output)_
```
Decode Steps | No Cache (ms) | With Cache (ms) | Speedup
-------------|---------------|-----------------|--------
20           | ___           | ___             | ___
40           | ___           | ___             | ___
80           | ___           | ___             | ___
```

---

## Summary Table (fill in once all experiments are done)

| # | Config | Val Loss | BLEU | Notes |
|---|---|---|---|---|
| 1 | Baseline (greedy) | — | — | |
| 2 | + Beam Search | — | — | decode-only change |
| 3 | + RoPE | — | — | retrained |
| 4 | + RMSNorm + SwiGLU | — | — | retrained |
| 5 | + Weight Tying | — | — | retrained |
| 6 | Flash Attention | — | — | same outputs, faster |
| 7 | KV Cache | — | — | same outputs, faster |

---

## What I'd Try Next (if I had more compute/time)

_(This section is for you — write down 2-3 ideas you didn't get to.
This is great interview material: "what would you try next" is a very
common closing question, and having a real answer beats improvising one.)_

- 
- 
- 
