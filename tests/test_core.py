import math
from pathlib import Path
import tempfile
import unittest

import torch

from generate import load_model
from model import GPTModel, Head, MultiHeadAttention
from utils import CharTokenizer


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def make_model(self):
        return GPTModel(
            vocab_size=11, n_embd=12, num_heads=3,
            n_layer=1, block_size=8, dropout=0.0,
        ).eval()

    def test_tokenizer_round_trip(self):
        text = 'ROMEO:\nHello, 世界!\n'
        tokenizer = CharTokenizer(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_head_shape(self):
        head = Head(12, 4, 8, dropout=0.0).eval()
        self.assertEqual(head(torch.randn(2, 5, 12)).shape, (2, 5, 4))

    def test_multihead_shape(self):
        attention = MultiHeadAttention(12, 3, 8, dropout=0.0).eval()
        self.assertEqual(attention(torch.randn(2, 5, 12)).shape, (2, 5, 12))

    @torch.no_grad()
    def test_causal_mask(self):
        x = torch.randn(2, 6, 12)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10
        for attention in (
            Head(12, 4, 8, dropout=0.0).eval(),
            MultiHeadAttention(12, 3, 8, dropout=0.0).eval(),
        ):
            with self.subTest(module=type(attention).__name__):
                torch.testing.assert_close(
                    attention(x)[:, :3], attention(changed)[:, :3],
                )

        model = self.make_model()
        idx = torch.randint(11, (2, 6))
        changed_idx = idx.clone()
        changed_idx[:, 3:] = (changed_idx[:, 3:] + 1) % 11
        logits, _ = model(idx)
        changed_logits, _ = model(changed_idx)
        torch.testing.assert_close(logits[:, :3], changed_logits[:, :3])

    def test_gpt_forward(self):
        model = self.make_model()
        idx = torch.randint(11, (2, 5))
        targets = torch.randint(11, (2, 5))
        logits, loss = model(idx)
        self.assertEqual(logits.shape, (2, 5, 11))
        self.assertIsNone(loss)
        flat_logits, loss = model(idx, targets)
        self.assertEqual(flat_logits.shape, (10, 11))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    @torch.no_grad()
    def test_checkpoint_round_trip(self):
        tokenizer = CharTokenizer('abcdefghijk')
        model = self.make_model()
        idx = torch.tensor([tokenizer.encode('abcde')])
        expected, _ = model(idx)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'test_checkpoint.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_stoi': tokenizer.stoi,
                'tokenizer_itos': tokenizer.itos,
                'vocab_size': tokenizer.vocab_size,
                'config': {
                    'n_embd': 12, 'num_heads': 3, 'n_layer': 1,
                    'block_size': 8, 'dropout': 0.0,
                },
            }, checkpoint_path)
            restored, restored_tokenizer = load_model(checkpoint_path)
            actual, _ = restored(idx)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(restored_tokenizer.stoi, tokenizer.stoi)
        self.assertEqual(restored_tokenizer.itos, tokenizer.itos)

    @torch.no_grad()
    def test_attention_matches_scaled_dot_product(self):
        # Unequal dimensions are essential to catch embedding-dimension scaling.
        head = Head(n_embd=12, head_size=4, block_size=8, dropout=0.0).double().eval()
        x = torch.randn(2, 5, 12, dtype=torch.float64)
        q, k, v = head.query(x), head.key(x), head.value(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        future = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)
        causal_mask = torch.zeros(5, 5, dtype=x.dtype).masked_fill(future, -torch.inf)
        expected = torch.softmax(scores + causal_mask, dim=-1) @ v
        torch.testing.assert_close(head(x), expected, rtol=1e-10, atol=1e-12)

    def test_invalid_head_configuration(self):
        for n_embd, num_heads in ((12, 0), (12, -1), (12, 5)):
            with self.subTest(n_embd=n_embd, num_heads=num_heads):
                with self.assertRaises(ValueError):
                    MultiHeadAttention(n_embd, num_heads, 8)

    def test_sequence_length_constraints(self):
        model = self.make_model()
        head = Head(12, 4, 8, dropout=0.0)
        for length in (0, 9):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    model(torch.zeros(2, length, dtype=torch.long))
                with self.assertRaises(ValueError):
                    head(torch.zeros(2, length, 12))
        # Both ends of the supported interval remain valid.
        for length in (1, 8):
            with self.subTest(valid_length=length):
                self.assertEqual(
                    model(torch.zeros(2, length, dtype=torch.long))[0].shape,
                    (2, length, 11),
                )

    def test_generation_parameter_constraints(self):
        model = self.make_model()
        idx = torch.zeros(1, 1, dtype=torch.long)
        for temperature in (0, -1, float('nan')):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    model.generate(idx, 1, temperature=temperature)
        for top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    model.generate(idx, 1, top_k=top_k)


if __name__ == '__main__':
    unittest.main()
