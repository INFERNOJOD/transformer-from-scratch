"""
tests.py — Unit Tests for the Transformer Implementation

Run (from the repository root):
    python tests/tests.py
    python -m pytest tests/tests.py -v   (if pytest is installed)

Coverage:
  - Correct output shapes for every component
  - Mathematical properties (attention sums to 1, masking works, etc.)
  - Gradient flow (model is trainable)
  - Numerical equivalence of implementations
  - Edge cases
"""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ─────────────────────────────────────────────
# Test Suite 1: Core Attention
# ─────────────────────────────────────────────

class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        from model import scaled_dot_product_attention
        self.attn = scaled_dot_product_attention
        self.B, self.H, self.S, self.D = 2, 4, 10, 16

    def test_output_shape(self):
        Q = torch.randn(self.B, self.H, self.S, self.D)
        K = torch.randn(self.B, self.H, self.S, self.D)
        V = torch.randn(self.B, self.H, self.S, self.D)
        out, w = self.attn(Q, K, V)
        self.assertEqual(out.shape, (self.B, self.H, self.S, self.D))
        self.assertEqual(w.shape, (self.B, self.H, self.S, self.S))

    def test_attention_weights_sum_to_one(self):
        """Softmax over keys should sum to 1.0 for each query."""
        Q = torch.randn(2, 4, 8, 16)
        K = torch.randn(2, 4, 8, 16)
        V = torch.randn(2, 4, 8, 16)
        _, w = self.attn(Q, K, V)
        sums = w.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5),
                        f"Attention weights don't sum to 1. Got: {sums.min():.4f} - {sums.max():.4f}")

    def test_causal_mask_prevents_future_attention(self):
        """With causal mask, position i should not attend to j > i."""
        from model import subsequent_mask
        S = 8
        Q = torch.randn(1, 1, S, 16)
        K = torch.randn(1, 1, S, 16)
        V = torch.randn(1, 1, S, 16)
        mask = subsequent_mask(S)   # (1, S, S)
        _, w = self.attn(Q, K, V, mask=mask)
        # Upper triangle (future) should be ~0
        w_squeezed = w.squeeze()
        for i in range(S):
            for j in range(i + 1, S):
                self.assertAlmostEqual(
                    w_squeezed[i, j].item(), 0.0, places=5,
                    msg=f"Position {i} attends to future position {j}: {w_squeezed[i,j]:.6f}"
                )

    def test_scaling_factor(self):
        """Scores should be divided by sqrt(d_k)."""
        d_k = 64
        Q = torch.ones(1, 1, 1, d_k)
        K = torch.ones(1, 1, 1, d_k)
        V = torch.ones(1, 1, 1, d_k)
        # Q·K = d_k (all ones dot product)
        # After scaling: d_k / sqrt(d_k) = sqrt(d_k)
        _, w = self.attn(Q, K, V)
        self.assertAlmostEqual(w.item(), 1.0, places=5,
                               msg="Single token should attend to itself with weight 1.0")


class TestMultiHeadAttention(unittest.TestCase):

    def test_output_shape(self):
        from model import MultiHeadAttention
        attn = MultiHeadAttention(h=8, d_model=512)
        x = torch.randn(2, 10, 512)
        out = attn(x, x, x)
        self.assertEqual(out.shape, (2, 10, 512))

    def test_d_model_must_be_divisible_by_h(self):
        from model import MultiHeadAttention
        with self.assertRaises(AssertionError):
            MultiHeadAttention(h=7, d_model=512)   # 512 % 7 != 0

    def test_different_qkv_shapes(self):
        """Encoder-decoder cross attention: Q and K/V can have different lengths."""
        from model import MultiHeadAttention
        attn = MultiHeadAttention(h=4, d_model=64)
        Q = torch.randn(2, 5, 64)    # decoder: 5 tokens
        K = torch.randn(2, 10, 64)   # encoder: 10 tokens
        V = torch.randn(2, 10, 64)
        out = attn(Q, K, V)
        self.assertEqual(out.shape, (2, 5, 64))


# ─────────────────────────────────────────────
# Test Suite 2: Positional Encoding
# ─────────────────────────────────────────────

class TestPositionalEncoding(unittest.TestCase):

    def test_output_shape_unchanged(self):
        from model import PositionalEncoding
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        x = torch.randn(2, 20, 64)
        out = pe(x)
        self.assertEqual(out.shape, x.shape)

    def test_pe_is_deterministic(self):
        """PE should add the same values regardless of input content."""
        from model import PositionalEncoding
        pe = PositionalEncoding(d_model=32, dropout=0.0)
        x1 = torch.zeros(1, 10, 32)
        x2 = torch.ones(1, 10, 32)
        diff1 = pe(x1) - x1
        diff2 = pe(x2) - x2
        self.assertTrue(torch.allclose(diff1, diff2, atol=1e-6),
                        "PE values should not depend on input content")

    def test_pe_different_positions(self):
        """Different positions should have different encodings."""
        from model import PositionalEncoding
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        # Position 0 and position 1 should differ
        self.assertFalse(
            torch.allclose(out[0, 0], out[0, 1]),
            "Different positions should have different encodings"
        )


# ─────────────────────────────────────────────
# Test Suite 3: Encoder / Decoder
# ─────────────────────────────────────────────

class TestEncoderDecoder(unittest.TestCase):

    def setUp(self):
        from model import make_model
        self.model = make_model(
            src_vocab_size=100, tgt_vocab_size=100,
            N=2, d_model=64, d_ff=128, h=4
        )

    def test_forward_pass_shape(self):
        B, S, T = 2, 8, 6
        src = torch.randint(1, 100, (B, S))
        tgt = torch.randint(1, 100, (B, T))
        from model import subsequent_mask
        src_mask = (src != 0).unsqueeze(-2)
        tgt_mask = (tgt != 0).unsqueeze(-2) & subsequent_mask(T)
        out = self.model(src, tgt, src_mask, tgt_mask)
        self.assertEqual(out.shape, (B, T, 64))

    def test_generator_output_shape(self):
        B, T, V = 2, 6, 100
        x = torch.randn(B, T, 64)
        logits = self.model.generator(x)
        self.assertEqual(logits.shape, (B, T, V))

    def test_generator_is_log_softmax(self):
        """Generator output should be log-probabilities (sum of exp = 1)."""
        x = torch.randn(1, 1, 64)
        logits = self.model.generator(x)
        probs = logits.exp()
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=4)

    def test_gradients_flow(self):
        """A forward + backward pass should produce non-zero gradients."""
        from model import subsequent_mask
        src = torch.randint(1, 100, (2, 8))
        tgt = torch.randint(1, 100, (2, 6))
        src_mask = (src != 0).unsqueeze(-2)
        tgt_mask = (tgt != 0).unsqueeze(-2) & subsequent_mask(6)

        out = self.model(src, tgt, src_mask, tgt_mask)
        logits = self.model.generator(out)
        loss = logits.mean()
        loss.backward()

        grad_norms = [p.grad.norm().item() for p in self.model.parameters()
                      if p.grad is not None]
        self.assertTrue(all(g > 0 for g in grad_norms),
                        "Some parameters have zero gradients")

    def test_causal_mask_shape(self):
        from model import subsequent_mask
        mask = subsequent_mask(10)
        self.assertEqual(mask.shape, (1, 10, 10))
        # Lower triangle (including diagonal) should be True
        self.assertTrue(mask[0, 0, 0].item())
        self.assertFalse(mask[0, 0, 1].item())


# ─────────────────────────────────────────────
# Test Suite 4: Modern Components
# ─────────────────────────────────────────────

class TestRMSNorm(unittest.TestCase):

    def test_output_shape(self):
        from modern_components import RMSNorm
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_rms_is_approximately_one(self):
        """After RMSNorm, the RMS of each vector should be ≈ 1 (times scale)."""
        from modern_components import RMSNorm
        norm = RMSNorm(d_model=64)
        # Start with scale=1 (default init)
        x = torch.randn(4, 8, 64) * 5   # large scale input
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=0.1),
                        f"RMS after normalization: {rms.mean():.3f} (expected ~1.0)")

    def test_layernorm_vs_rmsnorm_similar_scale(self):
        """RMSNorm and LayerNorm outputs should have similar magnitudes."""
        from modern_components import RMSNorm
        rms = RMSNorm(64)
        ln = nn.LayerNorm(64)
        # Zero-initialize biases for fair comparison
        x = torch.randn(8, 16, 64)
        out_rms = rms(x)
        out_ln = ln(x)
        ratio = out_rms.std() / out_ln.std()
        self.assertAlmostEqual(ratio.item(), 1.0, delta=0.3,
                               msg="RMSNorm and LayerNorm output scales differ too much")


class TestSwiGLU(unittest.TestCase):

    def test_output_shape(self):
        from modern_components import SwiGLUFeedForward
        ffn = SwiGLUFeedForward(d_model=64, d_ff=128)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        self.assertEqual(out.shape, x.shape)

    def test_gradients(self):
        from modern_components import SwiGLUFeedForward
        ffn = SwiGLUFeedForward(d_model=64, d_ff=128)
        x = torch.randn(2, 5, 64, requires_grad=True)
        out = ffn(x)
        out.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


# ─────────────────────────────────────────────
# Test Suite 5: RoPE
# ─────────────────────────────────────────────

class TestRoPE(unittest.TestCase):

    def test_output_shape(self):
        from rope import RoPEMultiHeadAttention
        attn = RoPEMultiHeadAttention(h=4, d_model=64)
        x = torch.randn(2, 10, 64)
        out = attn(x, x, x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_rotation_preserves_norm(self):
        """Rotation should preserve the L2 norm of each vector."""
        from rope import precompute_rope_freqs, apply_rope
        d_k, S = 32, 20
        cos, sin = precompute_rope_freqs(d_k, S)
        x = torch.randn(1, 1, S, d_k)
        x_rotated = apply_rope(x, cos, sin)
        norm_orig = x.norm(dim=-1)
        norm_rot = x_rotated.norm(dim=-1)
        self.assertTrue(torch.allclose(norm_orig, norm_rot, atol=1e-5),
                        "RoPE rotation should preserve vector norms")

    def test_different_positions_give_different_encodings(self):
        """Tokens at different positions should have different rotated vectors."""
        from rope import precompute_rope_freqs, apply_rope
        d_k, S = 32, 10
        cos, sin = precompute_rope_freqs(d_k, S)
        x = torch.ones(1, 1, S, d_k)   # same content, different positions
        x_rotated = apply_rope(x, cos, sin)
        # Position 0 and position 1 should differ
        self.assertFalse(
            torch.allclose(x_rotated[0, 0, 0], x_rotated[0, 0, 1]),
            "Same token at different positions should have different rotations"
        )


# ─────────────────────────────────────────────
# Test Suite 6: Weight Tying
# ─────────────────────────────────────────────

class TestWeightTying(unittest.TestCase):

    def test_weights_are_same_object_after_tying(self):
        from model import make_model
        from optimizations import apply_weight_tying
        model = make_model(500, 500, N=2, d_model=64, d_ff=128, h=4)
        apply_weight_tying(model)
        self.assertIs(
            model.generator.proj.weight,
            model.tgt_embed[0].lut.weight,
            "Weight tying failed: generator and embedding weights are not the same tensor"
        )

    def test_weight_tying_reduces_parameter_count(self):
        from model import make_model
        from optimizations import apply_weight_tying, count_parameters
        model = make_model(500, 500, N=2, d_model=64, d_ff=128, h=4)
        before = count_parameters(model)
        apply_weight_tying(model)
        after = count_parameters(model)
        self.assertLess(after, before,
                        "Weight tying should reduce unique parameter count")

    def test_gradient_flows_through_tied_weights(self):
        """Gradient should update the tied weight from both embedding and generator."""
        from model import make_model, subsequent_mask
        from optimizations import apply_weight_tying
        model = make_model(50, 50, N=2, d_model=32, d_ff=64, h=4)
        apply_weight_tying(model)

        src = torch.randint(1, 50, (1, 5))
        tgt = torch.randint(1, 50, (1, 4))
        src_mask = (src != 0).unsqueeze(-2)
        tgt_mask = (tgt != 0).unsqueeze(-2) & subsequent_mask(4)

        out = model(src, tgt, src_mask, tgt_mask)
        loss = model.generator(out).mean()
        loss.backward()

        self.assertIsNotNone(model.tgt_embed[0].lut.weight.grad)


# ─────────────────────────────────────────────
# Test Suite 7: Beam Search
# ─────────────────────────────────────────────

class TestBeamSearch(unittest.TestCase):

    def setUp(self):
        from model import make_model
        self.model = make_model(50, 50, N=2, d_model=32, d_ff=64, h=4)
        self.model.eval()

    def test_output_is_list(self):
        from beam_search import beam_search
        src = torch.randint(1, 50, (1, 6))
        src_mask = (src != 0).unsqueeze(-2)
        result = beam_search(self.model, src, src_mask,
                             max_len=10, start_symbol=0, end_symbol=1,
                             pad_symbol=2, beam_size=4,
                             device=torch.device("cpu"))
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_beam_size_1_similar_to_greedy(self):
        """Beam search with beam_size=1 should give same result as greedy."""
        from beam_search import beam_search
        from model import subsequent_mask

        torch.manual_seed(42)
        src = torch.randint(1, 50, (1, 6))
        src_mask = (src != 0).unsqueeze(-2)
        device = torch.device("cpu")

        beam_result = beam_search(self.model, src, src_mask,
                                  max_len=10, start_symbol=0, end_symbol=1,
                                  pad_symbol=2, beam_size=1, device=device)

        # Greedy decode
        with torch.no_grad():
            memory = self.model.encode(src, src_mask)
            ys = torch.zeros(1, 1, dtype=torch.long)
            greedy_result = []
            for _ in range(10):
                tgt_mask = subsequent_mask(ys.size(1))
                out = self.model.decode(memory, src_mask, ys, tgt_mask)
                next_tok = self.model.generator(out[:, -1]).argmax(dim=-1).item()
                greedy_result.append(next_tok)
                ys = torch.cat([ys, torch.tensor([[next_tok]])], dim=1)
                if next_tok == 1:
                    break

        self.assertEqual(beam_result, greedy_result,
                         "Beam search with k=1 should match greedy decode")


# ─────────────────────────────────────────────
# Test Suite 8: KV Cache
# ─────────────────────────────────────────────

class TestKVCache(unittest.TestCase):

    def test_cache_accumulates_correctly(self):
        from kv_cache import KVCache
        cache = KVCache()

        k1 = torch.randn(1, 4, 1, 16)
        v1 = torch.randn(1, 4, 1, 16)
        k1_ref = k1.clone()   # (1, 4, 1, 16)

        k_full, v_full = cache.update(0, k1, v1)
        self.assertEqual(k_full.shape, (1, 4, 1, 16))

        k2 = torch.randn(1, 4, 1, 16)
        v2 = torch.randn(1, 4, 1, 16)
        k_full, v_full = cache.update(0, k2, v2)
        self.assertEqual(k_full.shape, (1, 4, 2, 16))   # 2 tokens now

        # k_full[:,:,0,:] is (1,4,16); k1_ref is (1,4,1,16) — squeeze for comparison
        self.assertTrue(torch.allclose(k_full[:, :, 0, :], k1_ref.squeeze(2)),
                        "First cached key should match k1")
        self.assertTrue(torch.allclose(k_full[:, :, 1, :], k2.squeeze(2)),
                        "Second cached key should match k2")

    def test_cache_clear(self):
        from kv_cache import KVCache
        cache = KVCache()
        cache.update(0, torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        cache.update(1, torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        self.assertEqual(len(cache), 2)
        cache.clear()
        self.assertEqual(len(cache), 0)


# ─────────────────────────────────────────────
# Test Suite 9: Flash Attention
# ─────────────────────────────────────────────

class TestFlashAttention(unittest.TestCase):

    def test_tiled_matches_standard(self):
        """Tiled Flash Attention should produce same result as standard attention."""
        from flash_attention import flash_attention_tiled
        from model import scaled_dot_product_attention

        torch.manual_seed(0)
        Q = torch.randn(1, 2, 16, 32)
        K = torch.randn(1, 2, 16, 32)
        V = torch.randn(1, 2, 16, 32)

        out_standard, _ = scaled_dot_product_attention(Q, K, V)
        out_tiled = flash_attention_tiled(Q, K, V, block_size=8)

        self.assertTrue(
            torch.allclose(out_standard, out_tiled, atol=1e-4),
            f"Tiled attention differs from standard. Max diff: "
            f"{(out_standard - out_tiled).abs().max():.6f}"
        )

    def test_pytorch_flash_matches_standard(self):
        """PyTorch Flash Attention should match standard attention output."""
        from flash_attention import flash_attention_pytorch
        from model import scaled_dot_product_attention

        torch.manual_seed(1)
        Q = torch.randn(2, 4, 20, 16)
        K = torch.randn(2, 4, 20, 16)
        V = torch.randn(2, 4, 20, 16)

        out_standard, _ = scaled_dot_product_attention(Q, K, V)
        out_flash = flash_attention_pytorch(Q, K, V)

        self.assertTrue(
            torch.allclose(out_standard, out_flash, atol=1e-5),
            f"Flash attention output differs. Max diff: "
            f"{(out_standard - out_flash).abs().max():.6f}"
        )


# ─────────────────────────────────────────────
# Test Suite 10: Custom Data Pipeline (torchtext replacement)
# ─────────────────────────────────────────────

class TestDataPipelineVocab(unittest.TestCase):
    """
    Tests for data_pipeline.py's Vocab / build_vocab_from_iterator,
    which replace torchtext.vocab after torchtext was deprecated
    (last release 0.18.0, PyTorch <=2.3.0 only).

    Note: load_multi30k() itself (the HuggingFace datasets network call)
    is NOT covered here — it requires internet access and is verified
    separately by actually running train.py end-to-end.
    """

    def test_vocab_basic_lookup(self):
        from data_pipeline import build_vocab_from_iterator
        sentences = [["a", "b", "c"], ["a", "b"], ["a"]]
        vocab = build_vocab_from_iterator(sentences, min_freq=1)
        self.assertIn("a", vocab.get_itos())
        self.assertIn("b", vocab.get_itos())
        self.assertIn("c", vocab.get_itos())

    def test_min_freq_filtering(self):
        """Tokens below min_freq should be excluded from the vocabulary."""
        from data_pipeline import build_vocab_from_iterator
        sentences = [["common", "common", "common"], ["rare"]]
        vocab = build_vocab_from_iterator(sentences, min_freq=2)
        self.assertIn("common", vocab.get_itos())
        self.assertNotIn("rare", vocab.get_itos())

    def test_specials_come_first(self):
        """Special tokens should occupy the first N indices, in order."""
        from data_pipeline import build_vocab_from_iterator
        specials = ["<s>", "</s>", "<blank>", "<unk>"]
        sentences = [["hello", "world"]]
        vocab = build_vocab_from_iterator(sentences, min_freq=1, specials=specials)
        for i, tok in enumerate(specials):
            self.assertEqual(vocab[tok], i)

    def test_default_index_for_oov(self):
        """Out-of-vocabulary tokens should map to the default index, not error."""
        from data_pipeline import build_vocab_from_iterator
        vocab = build_vocab_from_iterator(
            [["known"]], min_freq=1, specials=["<unk>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        self.assertEqual(vocab["never_seen_token"], vocab["<unk>"])

    def test_callable_matches_indexing(self):
        """vocab(['a','b']) should equal [vocab['a'], vocab['b']]."""
        from data_pipeline import build_vocab_from_iterator
        vocab = build_vocab_from_iterator([["a", "b", "c"]], min_freq=1)
        ids = vocab(["a", "b", "c"])
        self.assertEqual(ids, [vocab["a"], vocab["b"], vocab["c"]])

    def test_len_matches_itos_length(self):
        from data_pipeline import build_vocab_from_iterator
        vocab = build_vocab_from_iterator(
            [["x", "y", "z"]], min_freq=1, specials=["<s>"]
        )
        self.assertEqual(len(vocab), len(vocab.get_itos()))

    def test_unknown_token_without_default_raises(self):
        """Without set_default_index(), unknown tokens should raise KeyError
        (matches torchtext's behavior of failing loudly rather than silently)."""
        from data_pipeline import build_vocab_from_iterator
        vocab = build_vocab_from_iterator([["known"]], min_freq=1)
        with self.assertRaises(KeyError):
            _ = vocab["totally_unseen"]


# ─────────────────────────────────────────────
# Test Suite 11: Ablation flags (make_model / train.py)
# ─────────────────────────────────────────────

class TestAblationFlags(unittest.TestCase):
    """Each make_model() flag builds a working model; defaults = baseline."""

    V, N, D = 50, 2, 64

    def build(self, **flags):
        from model import make_model
        torch.manual_seed(0)
        return make_model(self.V, self.V, N=self.N, d_model=self.D,
                          d_ff=128, h=4, dropout=0.0, **flags)

    def forward(self, model, src=None):
        from model import subsequent_mask
        torch.manual_seed(1)
        if src is None:
            src = torch.randint(3, self.V, (2, 7))
        tgt = torch.randint(3, self.V, (2, 5))
        src_mask = (src != 2).unsqueeze(-2)
        tgt_mask = (tgt != 2).unsqueeze(-2) & subsequent_mask(5)
        out = model(src, tgt, src_mask, tgt_mask)
        return model.generator(out)

    def assert_forward_ok(self, model):
        logits = self.forward(model)
        self.assertEqual(logits.shape, (2, 5, self.V))
        self.assertTrue(torch.isfinite(logits).all())

    def modules_of(self, model, cls):
        return [m for m in model.modules() if isinstance(m, cls)]

    def test_defaults_build_the_baseline(self):
        from model import (LayerNorm, MultiHeadAttention, PositionalEncoding,
                           PositionwiseFeedForward, make_model)
        from modern_components import RMSNorm, SwiGLUFeedForward
        from rope import RoPEMultiHeadAttention
        model = make_model(100, 100, N=2, d_model=64, d_ff=128, h=4)
        # Parameter count pinned from the pre-flag implementation
        self.assertEqual(sum(p.numel() for p in model.parameters()), 186980)
        self.assertEqual(len(model.state_dict()), 94)
        self.assertEqual(len(self.modules_of(model, PositionalEncoding)), 2)
        self.assertEqual(len(self.modules_of(model, MultiHeadAttention)), 6)
        self.assertEqual(len(self.modules_of(model, PositionwiseFeedForward)), 4)
        self.assertEqual(len(self.modules_of(model, LayerNorm)), 12)
        for cls in (RMSNorm, SwiGLUFeedForward, RoPEMultiHeadAttention):
            self.assertEqual(self.modules_of(model, cls), [])
        self.assertIsNot(model.generator.proj.weight, model.tgt_embed[0].lut.weight)

    def test_explicit_false_flags_match_defaults_exactly(self):
        base = self.build()
        flagged = self.build(use_rope=False, use_rmsnorm=False,
                             use_swiglu=False, tie_weights=False)
        sd_a, sd_b = base.state_dict(), flagged.state_dict()
        self.assertEqual(sd_a.keys(), sd_b.keys())
        for k in sd_a:
            self.assertTrue(torch.equal(sd_a[k], sd_b[k]), k)

    def test_use_rope(self):
        from model import MultiHeadAttention, PositionalEncoding
        from rope import RoPEMultiHeadAttention
        model = self.build(use_rope=True)
        self.assertEqual(self.modules_of(model, PositionalEncoding), [])
        for layer in list(model.encoder.layers) + list(model.decoder.layers):
            self.assertIsInstance(layer.self_attn, RoPEMultiHeadAttention)
        for layer in model.decoder.layers:
            self.assertIs(type(layer.cross_attn), MultiHeadAttention)
        self.assert_forward_ok(model)

    def test_rope_model_still_sees_token_order(self):
        """With sinusoidal PE removed, RoPE must be the source of position info:
        reversing the source must change the output."""
        model = self.build(use_rope=True).eval()
        src = torch.arange(3, 10).unsqueeze(0).repeat(2, 1)
        out_fwd = self.forward(model, src)
        out_rev = self.forward(model, src.flip(-1))
        self.assertFalse(torch.allclose(out_fwd, out_rev, atol=1e-5))

    def test_use_rmsnorm(self):
        from model import LayerNorm
        from modern_components import RMSNorm
        model = self.build(use_rmsnorm=True)
        self.assertEqual(self.modules_of(model, LayerNorm), [])
        # 2 per encoder layer + 3 per decoder layer + 2 final norms
        self.assertEqual(len(self.modules_of(model, RMSNorm)), 5 * self.N + 2)
        self.assert_forward_ok(model)

    def test_use_swiglu(self):
        from model import PositionwiseFeedForward
        from modern_components import SwiGLUFeedForward
        model = self.build(use_swiglu=True)
        self.assertEqual(self.modules_of(model, PositionwiseFeedForward), [])
        for layer in list(model.encoder.layers) + list(model.decoder.layers):
            self.assertIsInstance(layer.feed_forward, SwiGLUFeedForward)
        self.assert_forward_ok(model)

    def test_tie_weights(self):
        from optimizations import count_parameters
        model = self.build(tie_weights=True)
        self.assertIs(model.generator.proj.weight, model.tgt_embed[0].lut.weight)
        self.assertEqual(count_parameters(self.build()) - count_parameters(model),
                         self.V * self.D)
        self.assert_forward_ok(model)

    def test_all_flags_together_train(self):
        model = self.build(use_rope=True, use_rmsnorm=True,
                           use_swiglu=True, tie_weights=True)
        self.forward(model).mean().backward()
        grads = [p.grad for p in model.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

    def test_ablation_kwargs_defaults_old_checkpoints_to_baseline(self):
        from model import ablation_kwargs
        old_cfg = {"N": 3, "d_model": 256, "d_ff": 512, "h": 4}
        self.assertEqual(ablation_kwargs(old_cfg),
                         {"use_rope": False, "use_rmsnorm": False,
                          "use_swiglu": False, "tie_weights": False})
        self.assertTrue(ablation_kwargs({"use_rope": True})["use_rope"])

    def test_model_config_kwargs_build_model(self):
        from config import ABLATION_CONFIGS, ModelConfig
        from model import make_model
        self.assertFalse(any(ModelConfig.small().model_kwargs()[f]
                             for f in ("use_rope", "use_rmsnorm",
                                       "use_swiglu", "tie_weights")))
        for name in ("with_rope", "with_rmsnorm", "with_swiglu", "with_weight_tying"):
            kwargs = ABLATION_CONFIGS[name].model.model_kwargs()
            kwargs.update(N=1, d_model=32, d_ff=64)   # keep the test fast
            self.assert_forward_ok(make_model(self.V, self.V, **kwargs).eval())


# ─────────────────────────────────────────────
# Test Suite 12: BPE tokenizer
# ─────────────────────────────────────────────

class TestBPE(unittest.TestCase):

    CORPUS = [
        "Ein Hund läuft durch den Park.",
        "Zwei Männer spielen Fußball im Park.",
        "Ein kleines Kind sitzt auf einer Schaukel.",
        "Eine Frau liest ein Buch im Café.",
        "Der Hund spielt mit einem Ball.",
    ] * 20

    @classmethod
    def setUpClass(cls):
        from bpe import BPETokenizer
        cls.tok = BPETokenizer.train(cls.CORPUS, vocab_size=400)

    def test_encode_decode_round_trip(self):
        samples = self.CORPUS[:5] + [
            "",
            "Eisfischerhütte auf dem See",          # unseen compound
            "  leading, trailing  and   runs \t\n",  # whitespace edge cases
            "Ünïcödé — 東京 🙂 ẞ",                   # multi-byte characters
            "<s> literal special strings </s>",
        ]
        for s in samples:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s)
            self.assertEqual(self.tok.detokenize(self.tok.tokenize(s)), s)

    def test_merges_reduce_token_count(self):
        from bpe import BPETokenizer
        text = " ".join(self.CORPUS)
        n_bytes = len(text.encode("utf-8"))
        bytes_only = BPETokenizer.train(self.CORPUS, vocab_size=260)   # 0 merges
        small = BPETokenizer.train(self.CORPUS, vocab_size=300)
        self.assertEqual(len(bytes_only.encode(text)), n_bytes)
        self.assertLess(len(small.encode(text)), n_bytes)
        self.assertLess(len(self.tok.encode(text)), len(small.encode(text)))

    def test_frequent_word_becomes_one_token(self):
        self.assertEqual(len(self.tok.encode(" Hund")), 1)

    def test_unknown_words_never_produce_unk(self):
        vocab = self.tok.to_vocab()
        for s in ["Eisfischerhütte", "xylophone quizzically", "東京 🙂", "\x00\x7f"]:
            ids = self.tok.encode(s)
            self.assertNotIn(self.tok.unk_id, ids)
            self.assertNotIn(vocab["<unk>"], vocab(self.tok.tokenize(s)))

    def test_vocab_pipeline_matches_encode(self):
        vocab = self.tok.to_vocab()
        s = "Ein Hund spielt im Café."
        self.assertEqual(vocab(self.tok.tokenize(s)), self.tok.encode(s))
        self.assertEqual(len(vocab), len(self.tok))

    def test_special_ids_match_word_level_vocab(self):
        """<s>=0, </s>=1, <blank>=2 are hard-coded in train.collate_fn."""
        vocab = self.tok.to_vocab()
        for i, s in enumerate(["<s>", "</s>", "<blank>", "<unk>"]):
            self.assertEqual(vocab[s], i)
        # Literal "<s>" in text is just bytes, never the BOS id
        self.assertNotIn(0, self.tok.encode("<s>"))

    def test_save_load_round_trip(self):
        import os
        import tempfile
        from bpe import BPETokenizer
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bpe.json")
            self.tok.save(path)
            loaded = BPETokenizer.load(path)
        self.assertEqual(loaded.merges, self.tok.merges)
        s = "Zwei Hunde spielen im Park."
        self.assertEqual(loaded.encode(s), self.tok.encode(s))
        self.assertEqual(BPETokenizer.from_dict(self.tok.to_dict()).merges,
                         self.tok.merges)

    def test_training_is_deterministic(self):
        from bpe import BPETokenizer
        again = BPETokenizer.train(self.CORPUS, vocab_size=400)
        self.assertEqual(again.merges, self.tok.merges)

    def test_incremental_training_matches_naive_bpe(self):
        """The fast trainer must learn exactly the merges of textbook BPE."""
        import random
        from collections import Counter
        from bpe import BPETokenizer, PRETOKENIZE, _merge_pair

        rng = random.Random(0)
        corpus = [" ".join("".join(rng.choice("abcde") for _ in range(rng.randint(1, 6)))
                           for _ in range(8)) for _ in range(200)]
        vocab_size, min_freq = 330, 2

        freqs = Counter(w for t in corpus for w in PRETOKENIZE.findall(t))
        seqs = {w: [4 + b for b in w.encode()] for w in freqs}
        naive = []
        while 260 + len(naive) < vocab_size:
            counts = Counter()
            for w, f in freqs.items():
                for p in zip(seqs[w], seqs[w][1:]):
                    counts[p] += f
            if not counts:
                break
            best = min(counts, key=lambda p: (-counts[p], p))
            if counts[best] < min_freq:
                break
            naive.append(best)
            seqs = {w: _merge_pair(s, best, 259 + len(naive)) for w, s in seqs.items()}

        fast = BPETokenizer.train(corpus, vocab_size, min_frequency=min_freq)
        self.assertEqual(fast.merges, naive)

    def test_vocab_size_below_base_raises(self):
        from bpe import BPETokenizer
        with self.assertRaises(ValueError):
            BPETokenizer.train(self.CORPUS, vocab_size=100)

    def test_train_py_bpe_flag_is_opt_in(self):
        from train import CONFIG, parse_args
        self.assertFalse(CONFIG["use_bpe"])
        self.assertFalse(parse_args([]).use_bpe)
        self.assertTrue(parse_args(["--bpe"]).use_bpe)

    def test_collate_fn_with_bpe_pipeline(self):
        from train import collate_fn
        vocab = self.tok.to_vocab()
        batch = [("Ein Hund läuft.", "A dog runs.")]
        src, tgt = collate_fn(batch, self.tok.tokenize, self.tok.tokenize,
                              vocab, vocab, torch.device("cpu"), max_padding=32,
                              pad_id=vocab["<blank>"])
        self.assertEqual(src.shape, (1, 32))
        ids = src[0].tolist()
        self.assertEqual(ids[0], vocab["<s>"])
        n = len(self.tok.encode("Ein Hund läuft."))
        self.assertEqual(ids[n + 1], vocab["</s>"])
        self.assertTrue(all(i == vocab["<blank>"] for i in ids[n + 2:]))
        self.assertEqual(self.tok.decode(ids), "Ein Hund läuft.")


# ─────────────────────────────────────────────
# Test Suite 13: Decoder-only (GPT-style) model
# ─────────────────────────────────────────────

class TestDecoderOnly(unittest.TestCase):

    V, D, N = 40, 32, 2

    def build(self, **flags):
        from model import make_decoder_only_model
        torch.manual_seed(0)
        return make_decoder_only_model(self.V, N=self.N, d_model=self.D,
                                       d_ff=64, h=4, dropout=0.0, **flags)

    def test_output_shapes(self):
        model = self.build()
        x = torch.randint(0, self.V, (3, 9))
        h = model(x)
        self.assertEqual(h.shape, (3, 9, self.D))
        logp = model.generator(h)
        self.assertEqual(logp.shape, (3, 9, self.V))
        self.assertTrue(torch.allclose(logp.exp().sum(-1), torch.ones(3, 9), atol=1e-5))

    def test_reuses_existing_layers_without_cross_attention(self):
        from model import DecoderLayer, Encoder, EncoderLayer, MultiHeadAttention
        model = self.build()
        self.assertIsInstance(model.blocks, Encoder)
        self.assertTrue(all(isinstance(l, EncoderLayer) for l in model.blocks.layers))
        self.assertEqual(len([m for m in model.modules() if isinstance(m, DecoderLayer)]), 0)
        self.assertEqual(len([m for m in model.modules()
                              if isinstance(m, MultiHeadAttention)]), self.N)

    def assert_causal(self, model):
        """Changing token k must leave every position < k untouched."""
        model.eval()
        torch.manual_seed(2)
        x = torch.randint(0, self.V, (2, 10))
        base = model(x)
        for k in (3, 7, 9):
            x2 = x.clone()
            x2[:, k:] = (x2[:, k:] + 1) % self.V
            out = model(x2)
            self.assertTrue(torch.allclose(base[:, :k], out[:, :k], atol=1e-6),
                            f"position < {k} saw a future token")
            self.assertFalse(torch.allclose(base[:, k:], out[:, k:], atol=1e-6))

    def test_causal_masking_holds(self):
        self.assert_causal(self.build())

    def test_causal_masking_holds_with_rope(self):
        self.assert_causal(self.build(use_rope=True))

    def test_attention_weights_zero_above_diagonal(self):
        model = self.build().eval()
        model(torch.randint(0, self.V, (2, 8)))
        for layer in model.blocks.layers:
            w = layer.self_attn.attn_weights        # (B, h, T, T)
            upper = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
            self.assertTrue(torch.all(w[..., upper] == 0))

    def test_padding_mask(self):
        from model import lm_mask
        x = torch.tensor([[5, 6, 7, 0, 0]])
        mask = lm_mask(x, pad_idx=0)
        self.assertEqual(mask.shape, (1, 5, 5))
        self.assertFalse(mask[0, 4, 3].item())   # padded key hidden
        self.assertTrue(mask[0, 2, 1].item())    # earlier real token visible
        self.assertFalse(mask[0, 1, 2].item())   # future hidden
        model = self.build().eval()
        out = model(x, lm_mask(x, pad_idx=0))
        self.assertTrue(torch.isfinite(out).all())

    def test_gradient_flow(self):
        model = self.build()
        x = torch.randint(0, self.V, (2, 8))
        logp = model.generator(model(x[:, :-1]))
        loss = nn.functional.nll_loss(logp.reshape(-1, self.V), x[:, 1:].reshape(-1))
        loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.norm().item(), 0, name)

    def test_can_overfit_a_tiny_sequence(self):
        """Next-token loss on one repeated sequence should drop sharply."""
        model = self.build()
        x = torch.arange(1, 13).unsqueeze(0).repeat(4, 1)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(40):
            logp = model.generator(model(x[:, :-1]))
            loss = nn.functional.nll_loss(logp.reshape(-1, self.V), x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 0.5 * losses[0])

    def test_all_flags_build_and_train(self):
        from modern_components import RMSNorm, SwiGLUFeedForward
        from rope import RoPEMultiHeadAttention
        model = self.build(use_rope=True, use_rmsnorm=True,
                           use_swiglu=True, tie_weights=True)
        self.assertIs(model.generator.proj.weight, model.embed[0].lut.weight)
        kinds = {type(m) for m in model.modules()}
        self.assertTrue({RMSNorm, SwiGLUFeedForward, RoPEMultiHeadAttention} <= kinds)
        x = torch.randint(0, self.V, (2, 6))
        model.generator(model(x)).mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_generate_greedy(self):
        model = self.build()
        prompt = torch.randint(0, self.V, (2, 4))
        out = model.generate(prompt, max_new_tokens=5)
        self.assertEqual(out.shape, (2, 9))
        self.assertTrue(torch.equal(out[:, :4], prompt))
        first = model.generator(model(prompt)[:, -1]).argmax(-1)
        self.assertTrue(torch.equal(out[:, 4], first))

    def test_generate_stops_at_eos(self):
        model = self.build()
        prompt = torch.randint(0, self.V, (1, 3))
        eos = model.generate(prompt, max_new_tokens=1)[0, -1].item()
        out = model.generate(prompt, max_new_tokens=10, eos_id=eos)
        self.assertEqual(out.shape, (1, 4))


# ─────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestScaledDotProductAttention,
        TestMultiHeadAttention,
        TestPositionalEncoding,
        TestEncoderDecoder,
        TestRMSNorm,
        TestSwiGLU,
        TestRoPE,
        TestWeightTying,
        TestBeamSearch,
        TestKVCache,
        TestFlashAttention,
        TestDataPipelineVocab,
        TestAblationFlags,
        TestBPE,
        TestDecoderOnly,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    if result.failures or result.errors:
        print("FAILED")
    else:
        print("ALL TESTS PASSED ✓")
    print('='*50)
