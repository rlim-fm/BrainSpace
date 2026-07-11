import math
from abc import ABC
from math import comb
from itertools import combinations
from typing import Callable, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from torch import Tensor



class StaticNN(nn.Module, ABC):
    """
    Handles fixed length inputs and outputs
    """
    def __init__(self):
        super().__init__()
        self.output_layer = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass
    

class DynamicNN(nn.Module, ABC):
    """
    Handles variable length inputs and outputs.

    pool/seq2seq contract (enforced here for all subclasses):
      pool=False, seq2seq=False → final valid token  (B, H)
      pool=True,  seq2seq=False → mean over valid tokens  (B, H)
      pool=False, seq2seq=True  → all tokens  (B, S, H)
      pool=True,  seq2seq=True  → ValueError
    """
    def __init__(self, pool: bool = False, seq2seq: bool = False):
        super().__init__()
        if pool and seq2seq:
            raise ValueError("Cannot use pool=True with seq2seq=True")
        self.pool = pool
        self.seq2seq = seq2seq
        self.output_layer = None

    def select_output(self, rnn_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Route rnn_out (B, S, H) to the correct hidden representation.

        Args:
            rnn_out: (B, S, H) full sequence of hidden states
            lengths: (B,) number of valid timesteps per sequence
        """
        B = rnn_out.size(0)
        if self.pool:
            lengths_dev = lengths.to(rnn_out.device).unsqueeze(-1).clamp(min=1)
            return rnn_out.sum(dim=1) / lengths_dev
        elif self.seq2seq:
            return rnn_out
        else:
            last_idx = (lengths - 1).clamp(min=0).to(rnn_out.device)
            return rnn_out[torch.arange(B, device=rnn_out.device), last_idx]

    def forward(self, x: torch.Tensor, mask=Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pass

class MLP(StaticNN):
    """Simple fully-connected MLP for regression.

    Args:
        input_dim: dimensionality of input
        hidden_sizes: sequence of hidden layer sizes
        output_dim: dimensionality of output (1 for scalar regression)
        activation: activation class from torch.nn (default: nn.ReLU)
        dropout: dropout probability (default: 0.0)
    """

    def __init__(self,
                 input_dim: int = 1,
                 hidden_sizes: Sequence[int] = (64, 64),
                 activation = None,
                 output_dim: int = 1,
                 dropout: float = 0.0
                 ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(x)
        out = self.output_layer(hidden)
        return out, hidden

################################################################################
## 2. MODEL DEFINITIONS (Transformers & RNNs)
################################################################################

def adaptive_temperature_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Applies adaptive temperature softmax per head from softmax is not enough 
    (https://arxiv.org/abs/2410.01104).
    """
    original_probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-9), dim=-1, keepdim=True)

    # Polynomial coefficients for beta(θ) = 1 / θ
    # poly_fit corresponds to a 4th-degree polynomial: a0*x^4 + a1*x^3 + ... + a4
    poly_fit = torch.tensor([-0.037, 0.481, -2.3, 4.917, -1.791],
                            device=logits.device, dtype=logits.dtype)

    # Evaluate polynomial at entropy values via Horner's method
    beta = poly_fit[0]
    for coef in poly_fit[1:]:
        beta = beta * entropy + coef
    beta = torch.where(entropy > 0.5, torch.clamp(beta, min=1.0), torch.ones_like(entropy))

    # softmax with adaptive temperature
    return F.softmax(logits * beta, dim=-1)

class VanillaAttention(nn.Module):
    """Standard scaled dot-product attention with optional adaptive temperature softmax and masking."""
    def __init__(self, d_model, n_heads, aggregator='softmax', causal=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.aggregator = aggregator
        self.causal = causal
        
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear   = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)
        
        Q = Q.view(bsz, seq_len, self.n_heads, self.d_k).permute(0,2,1,3)
        K = K.view(bsz, seq_len, self.n_heads, self.d_k).permute(0,2,1,3)
        V = V.view(bsz, seq_len, self.n_heads, self.d_k).permute(0,2,1,3)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,S,S]
        
        # Apply masking if provided
        if mask is not None:
            # mask shape: (B, S) with True=valid, False=masked
            # Expand to (B, 1, 1, S) for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, S)
            # Set scores of masked positions to -inf so they become 0 after softmax
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool,
                                                device=scores.device))  # (S, S)
            scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # aggregator
        if self.aggregator == 'softmax':
            attn_weights = F.softmax(scores, dim=-1)
        elif self.aggregator == 'adaptive':
            attn_weights = adaptive_temperature_softmax(scores)
        else:
            raise ValueError(f"Unknown aggregator {self.aggregator}")
        
        # Replace NaN with 0 (result of softmax on -inf)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        context = torch.matmul(attn_weights, V)  # [B,H,S,Dk]
        context = context.permute(0,2,1,3).reshape(bsz, seq_len, d_model)
        output = self.out(context)
        return output


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            dim_ff: int = 128,
            dropout: float = 0.0,
            attn_factory: Optional[Callable] = None,
            pre_norm: bool = False,
            aggregator: str = 'softmax',
            causal: bool = False,
            *,
            device=None
    ):
        super().__init__()
        self.pre_norm = pre_norm
        # Attention layer. attn_factory(d_model, n_heads, causal) lets domain
        # packages substitute their own attention module; its forward may
        # return either a tensor or an (output, scores) tuple.
        if attn_factory is None:
            self.attn = VanillaAttention(d_model, n_heads, aggregator=aggregator, causal=causal)
        else:
            self.attn = attn_factory(d_model, n_heads, causal)
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _attend(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        out = self.attn(x, mask=mask)
        return out[0] if isinstance(out, tuple) else out

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention sublayer
        if self.pre_norm:
            attn_out = self._attend(self.norm1(x), mask)
            x = x + self.dropout(attn_out)
        else:
            attn_out = self._attend(x, mask)
            x = x + self.dropout(attn_out)
            x = self.norm1(x)

        # Feed-forward sublayer
        if self.pre_norm:
            x_norm = self.norm2(x)
            ff_out = self.ff(x_norm)
            x = x + self.dropout(ff_out)
        else:
            ff_out = self.ff(x)
            x = x + self.dropout(ff_out)
            x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            d_model: int = 32,
            n_heads: int = 2,
            num_layers: int = 1,
            dropout: float = 0.0,
            attn_factory: Optional[Callable] = None,
            pre_norm: bool = False,
            aggregator: str = 'softmax',
            causal: bool = False,
            *,
            device=None):
        super().__init__()
        self.pre_norm = pre_norm
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_ff=4*d_model,
                dropout=dropout,
                attn_factory=attn_factory,
                pre_norm=pre_norm,
                aggregator=aggregator,
                causal=causal
            )
            self.layers.append(block)

        self.last_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        if self.pre_norm:
            x = self.last_norm(x)
        return x

class SimpleTransformerModel(DynamicNN):
    def __init__(
                self,
                input_dim: int = 1,
                num_classes: int = 1,
                d_model: int = 32,
                n_heads: int = 2,
                num_layers: int = 1,
                dropout: float = 0.0,
                attn_factory: Optional[Callable] = None,
                pool: bool = False,
                pre_norm: bool = False,
                aggregator: str = 'softmax',
                causal: bool = True,
                *,
                device=None):
        super().__init__(pool=pool)
        self.input_linear = nn.Linear(input_dim, d_model)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            attn_factory=attn_factory,
            pre_norm=pre_norm,
            aggregator=aggregator,
            causal=causal
        )
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape
        lengths = mask.sum(dim=1).to(torch.int64) if mask is not None else \
                  torch.full((B,), S, dtype=torch.int64)
        x = torch.nan_to_num(x, nan=1e-7)
        x = self.input_linear(x)        # (B, S, d_model)
        x = self.encoder(x, mask=mask)  # (B, S, d_model)
        hidden = self.select_output(x, lengths)
        out = self.output_layer(hidden)
        return out, hidden

class MHA(SimpleTransformerModel):
    def __init__(self,
                 input_dim: int = 1,
                 num_classes: int = 1,
                 d_model: int = 32,
                 n_heads: int = 2,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 pool: bool = False,
                 pre_norm: bool = False,
                 aggregator: str = 'softmax',
                 causal: bool = True,
                 *,
                 device=None):
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            attn_factory=None,
            pool=pool,
            pre_norm=pre_norm,
            aggregator=aggregator,
            causal=causal,
            device=device
        )

class _BaseRNN(DynamicNN):
    """
    Base class for RNN variants (LSTM, GRU).
    Handles common logic for recurrent neural networks, with an overridable
    hidden-projection hook for domain-specific variants.
    """
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 pool: bool = False,
                 seq2seq: bool = False,
                 *,
                 device=None):
        super().__init__(pool=pool, seq2seq=seq2seq)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection
        self.input_linear = nn.Linear(input_dim, hidden_dim)

        # RNN layer - to be set by subclass
        self.rnn = self._create_rnn(hidden_dim, num_layers, dropout)

        # Optional projection applied to every hidden state (domain hook).
        # NOTE: constructed here, between self.rnn and self.output_layer, so
        # that subclasses adding parameterized projections preserve the exact
        # RNG consumption order of historical runs — do not move this line.
        self.hidden_proj = self._build_hidden_proj(hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def _create_rnn(self, hidden_dim, num_layers, dropout):
        """Create and return the RNN module. To be implemented by subclasses."""
        raise NotImplementedError

    def _build_hidden_proj(self, hidden_dim):
        """Optional nn.Module applied to the full hidden-state sequence before
        output selection; None (default) applies no projection. Domain
        subclasses override this (e.g. a max-plus projection layer)."""
        return None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through RNN.

        Args:
            x: input tensor of shape (B, S, input_dim) or (B, input_dim)
            mask: optional mask for padded positions (B, S) with True for valid positions

        Returns:
            output: tensor of shape (B, output_dim) or (B, S, output_dim) if seq2seq
            hidden: tensor of shape (B, hidden_dim) or (B, S, hidden_dim) if seq2seq
        """
        x = torch.nan_to_num(x, nan=1e-7)

        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, input_dim) -> (B, 1, input_dim)

        B, S, _ = x.shape

        # Project input
        x = self.input_linear(x)  # (B, S, hidden_dim)

        # Apply mask if provided (set masked positions to near-zero)
        if mask is not None:
            lengths = mask.sum(dim=1).to(torch.int64).to('cpu')
            x = rnn_utils.pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
        else:
            lengths = torch.full((B,), S, dtype=torch.int64, device='cpu')

        # RNN forward pass
        rnn_out, _ = self.rnn(x)

        if mask is not None:
            # Pad back to the full input length S (not just the batch's max valid
            # length) so the sequence dim is deterministic across micro-batches.
            rnn_out, _ = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True, total_length=S)  # (B, S, hidden_dim)

        # Apply the domain hidden-projection hook if present
        if self.hidden_proj is not None:
            rnn_out = self.hidden_proj(rnn_out)

        hidden = self.select_output(rnn_out, lengths)

        # Output projection
        output = self.output_layer(hidden)  # seq2seq: (B, S, output_dim), else (B, output_dim)

        return output, hidden


class LSTM(_BaseRNN):
    """
    LSTM (Long Short-Term Memory) network.

    Args:
        input_dim: dimensionality of input
        hidden_dim: dimensionality of hidden state
        output_dim: dimensionality of output
        num_layers: number of LSTM layers
        dropout: dropout probability
        pool: whether to pool hidden states (default: False)
        seq2seq: whether to return full sequence output (default: False)
        device: device for computation
    """
    def _create_rnn(self, hidden_dim, num_layers, dropout):
        return nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )


class GRU(_BaseRNN):
    """
    GRU (Gated Recurrent Unit) network.

    Args:
        input_dim: dimensionality of input
        hidden_dim: dimensionality of hidden state
        output_dim: dimensionality of output
        num_layers: number of GRU layers
        dropout: dropout probability
        pool: whether to pool hidden states (default: True)
        seq2seq: whether to return full sequence output (default: False)
        device: device for computation
    """
    def _create_rnn(self, hidden_dim, num_layers, dropout):
        return nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
