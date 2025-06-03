# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import uuid
import types
from kernels.sparse_gemv import SparseGEMV
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # hack import parent dir


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    sparsify: bool = False
    hist_path: str = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),

    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
}


def get_sparsity(matrix):
    spar = (matrix.numel() - matrix.count_nonzero()) / (matrix.numel())
    return spar * 100


def simulate_splitk(X, A, threshold, uuid):
    """Simulates the splitk gemv1 kernel"""

    N, Z = A.shape
    beam_width, seq_len, _ = X.shape

    Y = torch.empty(beam_width, seq_len, N,
                    device=X.device, dtype=torch.float16)
    mask = (X.abs() > threshold).float()
    masked = (X * mask).to(dtype=torch.float16)
    if seq_len == 1:
        for i in range(mask.shape[1]):
            input_tensor_c = masked.detach().clone().requires_grad_(False)
            torch.save(input_tensor_c, f"{uuid}.pt")
            masked_A = A * masked[0, i]
            # tensor_no_grad = masked_A.detach().clone().requires_grad_(False)
            # torch.save(tensor_no_grad, f"{uuid}.pt", _use_new_zipfile_serialization=True)

            # print(f"{uuid}: ", get_sparsity(masked_A))
            Y[0, i] = masked_A.T.sum(axis=0)
    else:
        for i in range(mask.shape[1]):
            masked_A = A * X[0, i]
            Y[0, i] = masked_A.T.sum(axis=0)

    return Y


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x),
                               freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


layer_uuid = f"{str(uuid.uuid4())}_0"
layer_count = -1


def _new_attn_forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
    global layer_uuid
    global layer_count
    if layer_count == 31:
        layer_count = 0
        layer_uuid = f"{str(uuid.uuid4())}_0"
    else:
        layer_count += 1
        layer_char_count = len(layer_uuid.split("_")[-1])
        layer_uuid = f"{layer_uuid[:-layer_char_count]}{layer_count}"
    print("Layer: ", layer_uuid)
    bsz, seqlen, _ = x.shape

    kv_size = self.n_local_heads * self.head_dim
    N, Z = self.wqkv.weight.shape
    beam_width, seq_len, _ = x.shape

    # self.wqkv needs to be split horizontally into three parts.
    q0, k0, v0 = self.wqkv.weight.split(kv_size)
    q = simulate_splitk(x, q0, self.thresh_q, layer_uuid + "_q0")
    k = simulate_splitk(x, k0, self.thresh_k, layer_uuid + "_k0")
    v = simulate_splitk(x, v0, self.thresh_v, layer_uuid + "_v0")

    # q,k,v = self.gemv1(x, self.wqkv.weight, self.thresh_q, self.thresh_k, self.thresh_v, self.sparsity_bin, kv_size).split([self.dim, kv_size, kv_size], dim=-1) # prefill logic taken care of in gemv

    q = q.view(bsz, seqlen, self.n_head, self.head_dim)
    k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

    q = apply_rotary_emb(q, freqs_cis)
    k = apply_rotary_emb(k, freqs_cis)

    q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

    if self.kv_cache is not None:
        k, v = self.kv_cache.update(input_pos, k, v)

    k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
    v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

    y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

    y = simulate_splitk(y, self.wo.weight, self.thresh_o, layer_uuid + "_y0") # prefill logic taken care of in gemv
    # y = self.gemv2(y, self.wo.weight, self.thresh_o, self.sparsity_bin) # prefill logic taken care of in gemv

    return y

class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def qkv_prefill(self, x: Tensor):
        pass

    def qkv_decode(self, x: Tensor):
        pass

    def apply_monkeypatch(self):
        self.old_forward = self.forward
        self.forward = types.MethodType(_new_attn_forward, self)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim


        # q,k,v = self.gemv1(x, self.wqkv.weight, self.thresh_q, self.thresh_k, self.thresh_v, self.sparsity_bin, kv_size).split([self.dim, kv_size, kv_size], dim=-1) # prefill logic taken care of in gemv
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1) # baseline

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        # y = self.gemv2(y, self.wo.weight, self.thresh_o, self.sparsity_bin) # prefill logic taken care of in gemv
        y = self.wo(y) # baseline

        return y

def _new_ffn_forward(self, x: Tensor) -> Tensor:

    N, Z = self.w1.weight.shape
    beam_width, seq_len, _ = x.shape

    # Adithya
    # This is the simulation logic for the ffn sparsification matrices.
    global layer_uuid

    w1 = simulate_splitk(x, self.w1.weight, self.thresh_gate, layer_uuid + "_w1")
    w1_silu = F.silu(w1)
    w3 = simulate_splitk(x, self.w3.weight, self.thresh_up, layer_uuid + "_w3")
    w2 = simulate_splitk(w1_silu * w3, self.w2.weight, self.thresh_down, layer_uuid + "_w2")
    # Adithya

    # w1 = self.gemv1(x, self.w1.weight, self.thresh_gate, self.sparsity_bin) # orig
    # w1_silu = F.silu(w1)
    # w3 = self.gemv1(x, self.w3.weight, self.thresh_up, self.sparsity_bin)
    # w2 = self.gemv2(w1_silu * w3, self.w2.weight, self.thresh_down, self.sparsity_bin)

    return w2

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        # w1, w1_silu, w3, w2
        # sparsity: self.ffn_sparsity = nn.Parameter(torch.tensor([0, 0, 0, 0]), requires_grad=False)

        self.sparsify = False

    def apply_monkeypatch(self):
        self.old_forward = self.forward
        self.forward = types.MethodType(_new_ffn_forward, self)

    def forward(self, x: Tensor) -> Tensor:
        # prefill logic taken care of in gemv
        # return self.gemv2(F.silu(self.gemv1(x, self.w1.weight, self.thresh_gate, self.sparsity_bin)) * self.gemv1(x, self.w3.weight, self.thresh_up, self.sparsity_bin), self.w2.weight, self.thresh_down, self.sparsity_bin) 
        return self.w2(F.silu(self.w1(x)) * self.w3(x)) # baseline


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.float16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
