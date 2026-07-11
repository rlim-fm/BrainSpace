"""Forward-contract invariants for model architectures."""
import pytest
import torch
import torch.nn as nn

from brainspace.models import (
    LSTM, GRU, MLP, SimpleTransformerModel, MHA, VanillaAttention, _BaseRNN,
)
from tests.conftest import requires_cuda


# ----------------------------------------------------------------------------
# RNN forward contracts (LSTM & GRU share _BaseRNN.forward)
# ----------------------------------------------------------------------------

@pytest.fixture(params=[LSTM, GRU])
def rnn_cls(request):
    return request.param


def test_rnn_pooled_output_shapes(rnn_cls):
    model = rnn_cls(input_dim=1, hidden_dim=8, output_dim=1, num_layers=1, device="cpu")
    x = torch.randn(4, 6, 1)
    out, hidden = model(x)
    assert out.shape == (4, 1)
    assert hidden.shape == (4, 8)


def test_rnn_2d_input_auto_unsqueezed(rnn_cls):
    model = rnn_cls(input_dim=3, hidden_dim=8, output_dim=2, num_layers=1, device="cpu")
    out, hidden = model(torch.randn(5, 3))  # (B, input_dim) -> seq len 1
    assert out.shape == (5, 2)
    assert hidden.shape == (5, 8)


def test_rnn_mask_forward_shapes(rnn_cls):
    model = rnn_cls(input_dim=1, hidden_dim=8, output_dim=1, num_layers=1, device="cpu")
    x = torch.randn(4, 6, 1)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[0, 4:] = False  # first sample shorter
    out, hidden = model(x, mask=mask)
    assert out.shape == (4, 1) and hidden.shape == (4, 8)


def test_rnn_seq2seq_output_shapes(rnn_cls):
    model = rnn_cls(input_dim=1, hidden_dim=8, output_dim=1, num_layers=1,
                    pool=False, seq2seq=True, device="cpu")
    x = torch.randn(4, 6, 1)
    out, hidden = model(x)
    assert out.shape == (4, 6, 1)
    assert hidden.shape == (4, 6, 8)


def test_rnn_seq2seq_requires_pool_false(rnn_cls):
    with pytest.raises(ValueError):
        rnn_cls(input_dim=1, hidden_dim=8, pool=True, seq2seq=True, device="cpu")


def test_rnn_backward_populates_grads(rnn_cls):
    model = rnn_cls(input_dim=1, hidden_dim=8, output_dim=1, num_layers=1, device="cpu")
    out, _ = model(torch.randn(4, 6, 1))
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)


# ----------------------------------------------------------------------------
# MLP & Transformer
# ----------------------------------------------------------------------------

def test_mlp_forward_shapes():
    model = MLP(input_dim=3, hidden_sizes=(8, 8), output_dim=2)
    out, hidden = model(torch.randn(5, 3))
    assert out.shape == (5, 2) and hidden.shape == (5, 8)


def test_transformer_pooled_and_masked_shapes():
    model = SimpleTransformerModel(input_dim=1, num_classes=1, d_model=8,
                                   n_heads=2, num_layers=1, pool=True)
    x = torch.randn(4, 6, 1)
    out, hidden = model(x)
    assert out.shape == (4, 1) and hidden.shape == (4, 8)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[0, 4:] = False
    out_m, _ = model(x, mask=mask)
    assert out_m.shape == (4, 1)


# ----------------------------------------------------------------------------
# CUDA
# ----------------------------------------------------------------------------

@requires_cuda
def test_rnn_forward_on_cuda(rnn_cls):
    model = rnn_cls(input_dim=1, hidden_dim=8, output_dim=1, device="cuda").cuda()
    out, hidden = model(torch.randn(4, 6, 1, device="cuda"))
    assert out.is_cuda and out.shape == (4, 1)


# ----------------------------------------------------------------------------
# Domain extension hooks
# ----------------------------------------------------------------------------

def test_base_rnn_hidden_proj_hook_applied():
    """_build_hidden_proj (domain hook) output is applied to every hidden state."""

    class DoublingLSTM(LSTM):
        def _build_hidden_proj(self, hidden_dim):
            class Doubler(nn.Module):
                def forward(self, x):
                    return 2 * x
            return Doubler()

    torch.manual_seed(0)
    plain = LSTM(input_dim=1, hidden_dim=8, output_dim=1, num_layers=1)
    torch.manual_seed(0)
    doubled = DoublingLSTM(input_dim=1, hidden_dim=8, output_dim=1, num_layers=1)
    x = torch.randn(4, 6, 1)
    _, h_plain = plain(x)
    _, h_doubled = doubled(x)
    assert torch.allclose(h_doubled, 2 * h_plain, atol=1e-6)


def test_base_rnn_default_has_no_hidden_proj():
    model = LSTM(input_dim=1, hidden_dim=8, output_dim=1)
    assert model.hidden_proj is None


def test_transformer_attn_factory_substitutes_attention():
    """attn_factory replaces VanillaAttention in every block; tuple-returning
    attention modules are unwrapped transparently."""
    calls = []

    class RecordingAttention(VanillaAttention):
        def forward(self, x, mask=None):
            calls.append(x.shape)
            return super().forward(x, mask=mask), None  # tuple form

    model = SimpleTransformerModel(
        input_dim=1, num_classes=1, d_model=8, n_heads=2, num_layers=2,
        attn_factory=lambda d, h, causal: RecordingAttention(d, h, causal=causal),
    )
    assert all(isinstance(b.attn, RecordingAttention) for b in model.encoder.layers)
    out, hidden = model(torch.randn(3, 5, 1))
    assert out.shape == (3, 1)
    assert len(calls) == 2
