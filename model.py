"""GPT with a `mixer` switch between Gated DeltaNet (GDN) and softmax attention.

Both paths share: RMSNorm pre-norm, SwiGLU MLP, tied embeddings, no biases,
depth-scaled c_proj/o_proj init, FusedLinearCrossEntropyLoss.

Mixers:
- `mixer='gdn'` (default) → fla.layers.gated_deltanet.GatedDeltaNet
  Linear-attention recurrence with delta-rule updates and per-head Mamba2-style
  forget gate. Implicit causality, no positional embedding. Reference:
  https://arxiv.org/abs/2412.06464
- `mixer='softmax'` → CausalSelfAttention with RoPE
  Standard MHA via F.scaled_dot_product_attention. Uses `n_head` and `rope_theta`.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.modules import FusedLinearCrossEntropyLoss


def _precompute_rope_freqs(head_dim: int, seq_len: int, theta: float, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [T, head_dim/2]
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]. Rotates adjacent channel pairs (x[..., 2i], x[..., 2i+1]).
    T = x.shape[-2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, :T, :].to(x.dtype)
    s = sin[None, None, :T, :].to(x.dtype)
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        cos, sin = _precompute_rope_freqs(self.head_dim, config.block_size, config.rope_theta)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, self.rope_cos, self.rope_sin)
        k = _apply_rope(k, self.rope_cos, self.rope_sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """SwiGLU MLP: c_fc outputs (gate || up); down-projection is c_proj."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 2 * config.ffn_intermediate_size, bias=False)
        self.c_proj = nn.Linear(config.ffn_intermediate_size, config.n_embd, bias=False)

    def forward(self, x):
        gate, up = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(F.silu(gate) * up)


class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        if config.mixer == 'gdn':
            self.attn = GatedDeltaNet(
                hidden_size=config.n_embd,
                head_dim=config.gdn_head_dim,
                num_heads=config.gdn_num_heads,
                num_v_heads=config.gdn_num_v_heads,
                expand_v=config.gdn_expand_v,
                mode=config.gdn_mode,
                use_gate=config.gdn_use_gate,
                use_short_conv=config.gdn_use_short_conv,
                conv_size=config.gdn_conv_size,
                conv_bias=False,
                allow_neg_eigval=config.gdn_allow_neg_eigval,
                layer_idx=layer_idx,
                norm_eps=config.norm_eps,
            )
        elif config.mixer == 'softmax':
            self.attn = CausalSelfAttention(config)
        else:
            raise ValueError(f"Unknown mixer={config.mixer!r}; expected 'gdn' or 'softmax'.")
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        if isinstance(self.attn, GatedDeltaNet):
            attn_out, _, _ = self.attn(self.ln_1(x))
        else:
            attn_out = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


def _default_ffn_intermediate(n_embd: int) -> int:
    # 8/3 × n_embd rounded up to next multiple of 64 (LLaMA / lm-engine convention).
    return int(math.ceil(8 * n_embd / 3 / 64)) * 64


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_embd: int = 768
    ffn_intermediate_size: int = None  # None → 8/3·n_embd rounded to multiple of 64
    norm_eps: float = 1e-5
    # Which sequence mixer to use.
    mixer: str = 'gdn'  # 'gdn' or 'softmax'
    # GDN sequence-mixer config (used when mixer='gdn')
    gdn_head_dim: int = 128
    gdn_num_heads: int = 6
    gdn_num_v_heads: int = None      # None → defaults to num_heads inside GDN
    gdn_expand_v: float = 1.0
    gdn_mode: str = "chunk"
    gdn_use_gate: bool = False
    gdn_use_short_conv: bool = True
    gdn_conv_size: int = 4
    gdn_allow_neg_eigval: bool = False
    # Softmax-attention config (used when mixer='softmax')
    n_head: int = 12
    rope_theta: float = 10000.0


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        if config.ffn_intermediate_size is None:
            config.ffn_intermediate_size = _default_ffn_intermediate(config.n_embd)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f=nn.RMSNorm(config.n_embd, eps=config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Tied embeddings (already standard in nanoGPT)
        self.transformer.wte.weight = self.lm_head.weight

        # Fused linear + cross-entropy (avoids materializing the (B·T, V) logits tensor,
        # which at our 8×4096×100352 fp32 shape is ~13 GB and OOMs activation memory).
        self.loss_fn = FusedLinearCrossEntropyLoss(ignore_index=-1)

        self.apply(self._init_weights)
        # Depth-scaled re-init for output projections (Block.attn.{c,o}_proj, Block.mlp.c_proj).
        depth_scale = 0.02 / math.sqrt(2 * config.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=depth_scale)

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        # Tied embeddings → wte.weight and lm_head.weight share storage and are
        # counted once by parameters().
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, GatedDeltaNet):
            # A_log / dt_bias keep GDN's bespoke init (set in GatedDeltaNet.__init__).
            # Linear children (q/k/v/a/b/o_proj) are caught by the nn.Linear branch.
            pass

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Fused path: never materializes the full [B·T, V] logits tensor.
            loss = self.loss_fn(x, targets, self.lm_head.weight)
            return None, loss
        # inference-time mini-opt: only forward the lm_head on the last position
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.no_grad()
    def logits_at(self, idx, positions):
        """Eval helper: per-position logits at `positions` (1D LongTensor of time
        indices, shared across the batch). Returns [B, len(positions), vocab].
        Used to score answer-position recall accuracy without materializing the
        full [B, T, vocab] logits tensor."""
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x[:, positions, :])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, eps=1e-8):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # GDN marks A_log and dt_bias with `_no_weight_decay = True` (see fla.layers.gated_deltanet).
        # Honor that flag here so they end up in the no-decay group regardless of dim.
        decay_params, nodecay_params = [], []
        for n, p in param_dict.items():
            if getattr(p, '_no_weight_decay', False) or p.dim() < 2:
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args)
        print(f"using fused AdamW: {use_fused} (eps={eps})")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """MFU in units of H100 SXM bf16 dense peak (989 TFLOPS).

        Both mixers use the 6N approximation; for softmax attention the T²·H·Q·d
        correction is dropped (matches baseline practice at these scales).
        """
        N = self.get_num_params()
        T = self.config.block_size
        flops_per_token = 6 * N
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 989e12  # H100 SXM bf16 dense
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
