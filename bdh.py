# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        # Current attention
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)

        # actually helps with training
        x = self.ln(x)  # B, 1, T, D

        for level in range(C.n_layer):
            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



# bdh-gpu
@dataclasses.dataclass
class BDH_GPUConfig:
    n_layer: int = 6           # L in paper
    n_embd: int = 256          # D in paper
    n_head: int = 4            # H in paper
    n_neurons: int = 32768     # N in paper
    dropout: float = 0.05
    vocab_size: int = 256
    rope_theta: float = 2**16


class LinearAttention(nn.Module):
    """Linear Attention with RoPE (causal)."""
    
    def __init__(self, config: BDH_GPUConfig):
        super().__init__()
        self.config = config
        N_per_head = config.n_neurons // config.n_head
        self.freqs = nn.Buffer(
            get_freqs(N_per_head, theta=config.rope_theta, dtype=torch.float32)
            .view(1, 1, 1, N_per_head)
        )

    @staticmethod
    def rope(phases, v):
        """Apply rotary position embeddings."""
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        """
        Args:
            Q, K: B, H, T, N//H (sparse activations)
            V: B, 1, T, D (dense representation)
        Returns:
            B, H, T, D
        """
        _, _, T, _ = Q.size()
        
        # Compute position-dependent phases
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        
        Qr = self.rope(r_phases, Q)
        Kr = self.rope(r_phases, K)
        
        # Causal linear attention: (Q @ K^T) with causal mask, then @ V
        scores = (Qr @ Kr.mT).tril(diagonal=-1)
        return scores @ V


class BDH_GPU(nn.Module):
    """
    BDH-GPU (Definition 4) implementation.
    
    Architecture per layer:
        x = relu(v* @ decoder_x)           # project to sparse
        a* = LinearAttn(Q=x, K=x, V=v*)    # attend in sparse space
        y = relu(ln(a*) @ decoder_y) * x   # gated sparse output
        v* = v* + ln(y @ encoder)          # project back to dense
    """
    
    def __init__(self, config: BDH_GPUConfig):
        super().__init__()
        self.config = config
        
        D = config.n_embd
        H = config.n_head
        N = config.n_neurons
        
        # Layer norm (no learnable params as per paper)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        
        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, D)
        
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Projections (paper naming convention)
        # decoder_x: D -> N//H per head (expand to sparse)
        self.decoder_x = nn.Parameter(
            torch.zeros((H, D, N // H)).normal_(std=0.02)
        )
        # decoder_y: D -> N//H per head (second sparse projection)
        self.decoder_y = nn.Parameter(
            torch.zeros((H, D, N // H)).normal_(std=0.02)
        )
        # encoder: N -> D (compress from sparse back to dense)
        self.encoder = nn.Parameter(
            torch.zeros((N, D)).normal_(std=0.02)
        )
        
        # Output projection
        self.readout = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )
        
        # Linear attention module
        self.attn = LinearAttention(config)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: B, T token indices
            targets: B, T target indices (optional, for loss)
        Returns:
            logits: B, T, vocab_size
            loss: scalar if targets provided, else None
        """
        config = self.config
        B, T = idx.size()
        N = config.n_neurons
        L = config.n_layer
        
        # Embed and add head dimension: B, 1, T, D
        v_ast = self.ln(self.wte(idx).unsqueeze(1))
        
        for _ in range(L):
            # Project to sparse space: B, H, T, N//H
            x = F.relu(v_ast @ self.decoder_x)
            
            # Linear attention with sparse Q,K and dense V
            a_ast = self.attn(Q=x, K=x, V=v_ast)  # B, H, T, D
            
            # Gated sparse output: B, H, T, N//H
            y = F.relu(self.ln(a_ast) @ self.decoder_y) * x
            
            # Reshape for encoder projection: B, 1, T, N
            y = y.transpose(1, 2).reshape(B, 1, T, N)
            y = self.drop(y)
            
            # Project back to dense and residual: B, 1, T, D
            v_ast = v_ast + self.ln(y @ self.encoder)
        
        # Final layer norm
        v_ast = self.ln(v_ast)
        
        # Output logits: B, T, vocab_size
        logits = v_ast.squeeze(1) @ self.readout
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx