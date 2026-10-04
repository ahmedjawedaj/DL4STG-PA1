"""Compact Autoformer (Wu et al., 2021) written from scratch for Task 2.

Structure follows the paper (Sections 3.1-3.2) and the public reference implementation
github.com/thuml/Autoformer (layers/Autoformer_EncDec.py, layers/AutoCorrelation.py,
models/Autoformer.py), which was read and adapted, not copied. Differences from the reference:
  * delay scoring and delay aggregation reuse the Question 1 convention
      R(tau) = sum_t q[t] k[(t - tau) mod L],   z[t] = sum_j alpha_j v[(t - tau_j) mod L]
    with per-example top-k delays in training and inference (the reference shares delays
    across the batch during training);
  * the "time-feature" (mark) embedding carries whatever known-in-advance inputs a run uses:
    index-derived phase features and, in the "future" covariate mode, the optional variables;
  * the trend path is univariate (c_out = 1) even when covariates enter as values.

Both distinguishing mechanisms are present in every block:
  (a) progressive series decomposition (SeriesDecomp after every sublayer, trend accumulated
      through the decoder), and
  (b) Auto-Correlation (FFT delay scores + time-delay aggregation) in place of dot-product
      attention, for encoder self, decoder self and decoder-encoder cross mixing.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SeriesDecomp(nn.Module):
    """Centered moving average with replicate padding; returns (seasonal, trend)."""

    def __init__(self, kernel: int):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        self.kernel = kernel

    def forward(self, x):                                   # [B, L, C]
        half = self.kernel // 2
        padded = torch.cat([x[:, :1].expand(-1, half, -1), x,
                            x[:, -1:].expand(-1, half, -1)], 1)
        trend = F.avg_pool1d(padded.transpose(1, 2), self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend


def delay_scores(q, k):
    """q, k: [B, H, E, L] -> R(tau) for tau = 0..L-1, averaged over heads/features: [B, L]."""
    q = q - q.mean(-1, keepdim=True)
    k = k - k.mean(-1, keepdim=True)
    spec = torch.fft.rfft(q, dim=-1) * torch.fft.rfft(k, dim=-1).conj()
    return torch.fft.irfft(spec, n=q.shape[-1], dim=-1).mean((1, 2))


def aggregate_delays(v, delays, weights):
    """v: [B, H, E, L]; delays, weights: [B, K] -> sum_j w_j * roll(v, +tau_j): [B, H, E, L]."""
    b, h, e, length = v.shape
    kk = delays.shape[-1]
    t = torch.arange(length, device=v.device)
    src = (t.view(1, 1, length) - delays.long().unsqueeze(-1)) % length          # [B, K, L]
    src = src.view(b, 1, 1, kk, length).expand(b, h, e, kk, length)
    shifted = torch.gather(v.unsqueeze(3).expand(-1, -1, -1, kk, -1), -1, src)
    return (shifted * weights.view(b, 1, 1, kk, 1)).sum(3)


class AutoCorrelation(nn.Module):
    """Period-based dependency discovery + time-delay aggregation (paper Sec. 3.2)."""

    def __init__(self, d_model: int, heads: int, factor: float = 1.0, min_delay: int = 1):
        super().__init__()
        self.heads, self.factor, self.min_delay = heads, factor, min_delay
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)
        self.last_delays = None

    def forward(self, queries, keys, values):              # [B, L, d], [B, S, d], [B, S, d]
        b, length, d = queries.shape
        s = keys.shape[1]
        if s < length:                                      # reference: zero-pad keys/values
            pad = queries.new_zeros(b, length - s, d)
            keys, values = torch.cat([keys, pad], 1), torch.cat([values, pad], 1)
        else:
            keys, values = keys[:, :length], values[:, :length]
        shape = (b, length, self.heads, d // self.heads)
        q = self.q(queries).view(shape).permute(0, 2, 3, 1)  # [B, H, E, L]
        k = self.k(keys).view(shape).permute(0, 2, 3, 1)
        v = self.v(values).view(shape).permute(0, 2, 3, 1)
        scores = delay_scores(q, k)                          # [B, L]
        top_k = max(1, int(self.factor * math.log(length)))
        band = scores[:, self.min_delay:]
        weights, idx = band.topk(top_k, dim=-1)
        delays = idx + self.min_delay
        self.last_delays = delays.detach()
        mixed = aggregate_delays(v, delays, weights.softmax(-1))
        return self.out(mixed.permute(0, 3, 1, 2).reshape(b, length, d))


class SeasonalLayerNorm(nn.Module):
    """LayerNorm followed by removal of the temporal mean (reference: my_Layernorm)."""

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(1, keepdim=True)


class DataEmbedding(nn.Module):
    """Value (token) embedding + known-input (mark) embedding, no positional encoding."""

    def __init__(self, c_in, mark_dim, d_model, dropout, mark_kernel=1, pad="circular",
                 mark_mlp=0):
        super().__init__()
        self.value = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode=pad, bias=False)
        # mark_kernel=1 is the reference linear time-feature embedding; >1 lets each step see
        # its neighbours' known inputs (zero padding: no wrap-around between horizon ends)
        self.mark = (nn.Conv1d(mark_dim, d_model, mark_kernel, padding=mark_kernel // 2,
                               bias=False) if mark_dim else None)
        # optional nonlinear covariate encoder: a per-step two-layer MLP on the known inputs,
        # so combinations of covariates (not only their sum) can shape the embedding
        self.mark_mlp = (nn.Sequential(nn.Linear(mark_dim, mark_mlp), nn.GELU(),
                                       nn.Linear(mark_mlp, d_model, bias=False))
                         if mark_dim and mark_mlp else None)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mark):
        out = self.value(x.transpose(1, 2)).transpose(1, 2)
        if self.mark is not None:
            out = out + self.mark(mark.transpose(1, 2)).transpose(1, 2)
        if self.mark_mlp is not None:
            out = out + self.mark_mlp(mark)
        return self.drop(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, kernel, dropout, factor):
        super().__init__()
        self.attn = AutoCorrelation(d_model, heads, factor)
        self.ff = nn.Sequential(nn.Conv1d(d_model, d_ff, 1, bias=False), nn.GELU(),
                                nn.Dropout(dropout), nn.Conv1d(d_ff, d_model, 1, bias=False))
        self.decomp1, self.decomp2 = SeriesDecomp(kernel), SeriesDecomp(kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.decomp1(x + self.drop(self.attn(x, x, x)))
        y = self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2))
        x, _ = self.decomp2(x + y)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, kernel, dropout, factor, c_out, pad="circular"):
        super().__init__()
        self.self_attn = AutoCorrelation(d_model, heads, factor)
        self.cross_attn = AutoCorrelation(d_model, heads, factor)
        self.ff = nn.Sequential(nn.Conv1d(d_model, d_ff, 1, bias=False), nn.GELU(),
                                nn.Dropout(dropout), nn.Conv1d(d_ff, d_model, 1, bias=False))
        self.decomp1, self.decomp2, self.decomp3 = (SeriesDecomp(kernel) for _ in range(3))
        self.trend_proj = nn.Conv1d(d_model, c_out, 3, padding=1, padding_mode=pad, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cross):
        x, t1 = self.decomp1(x + self.drop(self.self_attn(x, x, x)))
        x, t2 = self.decomp2(x + self.drop(self.cross_attn(x, cross, cross)))
        y = self.drop(self.ff(x.transpose(1, 2)).transpose(1, 2))
        x, t3 = self.decomp3(x + y)
        trend = self.trend_proj((t1 + t2 + t3).transpose(1, 2)).transpose(1, 2)
        return x, trend


class Autoformer(nn.Module):
    def __init__(self, enc_in, mark_dim, seq_len=168, label_len=48, pred_len=168, d_model=32,
                 heads=4, d_ff=64, e_layers=1, d_layers=1, kernel=25, dropout=0.1, factor=1.0,
                 mark_kernel=1, pad="circular", trend_init="mean", mark_mlp=0, cov_head=0):
        super().__init__()
        self.trend_init = trend_init
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        self.decomp = SeriesDecomp(kernel)
        self.enc_embed = DataEmbedding(enc_in, mark_dim, d_model, dropout, mark_kernel, pad,
                                       mark_mlp)
        self.dec_embed = DataEmbedding(enc_in, mark_dim, d_model, dropout, mark_kernel, pad,
                                       mark_mlp)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, kernel, dropout, factor)
                                      for _ in range(e_layers)])
        self.enc_norm = SeasonalLayerNorm(d_model)
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, kernel, dropout, factor, 1, pad)
                                      for _ in range(d_layers)])
        self.dec_norm = SeasonalLayerNorm(d_model)
        self.projection = nn.Linear(d_model, 1)
        # optional direct covariate head: a per-hour MLP from the known inputs of each horizon
        # step straight to that step's output, added to the Autoformer forecast. The
        # decomposition and Auto-Correlation paths are unchanged; they model what the
        # covariates alone do not explain.
        self.cov_head = (nn.Sequential(nn.Linear(mark_dim, cov_head), nn.GELU(),
                                       nn.Linear(cov_head, cov_head), nn.GELU(),
                                       nn.Linear(cov_head, 1))
                         if mark_dim and cov_head else None)

    def forward(self, x_enc, mark_enc, mark_dec):
        """x_enc [B, seq_len, enc_in] (channel 0 = target); mark_enc [B, seq_len, m];
        mark_dec [B, label_len + pred_len, m]. Returns the target forecast [B, pred_len]."""
        target = x_enc[..., :1]
        # reference: the horizon part of the trend starts at the window mean. "last" starts it
        # at the last observed value, so the smooth decoder trend only has to model the
        # departure from persistence rather than jump back from the mean.
        start = target.mean(1, keepdim=True) if self.trend_init == "mean" else target[:, -1:]
        mean = start.expand(-1, self.pred_len, -1)
        seasonal, trend = self.decomp(x_enc)
        trend_init = torch.cat([trend[:, -self.label_len:, :1], mean], 1)
        seasonal_init = torch.cat([seasonal[:, -self.label_len:],
                                   x_enc.new_zeros(x_enc.shape[0], self.pred_len,
                                                   x_enc.shape[2])], 1)
        enc = self.enc_embed(x_enc, mark_enc)
        for layer in self.encoder:
            enc = layer(enc)
        enc = self.enc_norm(enc)
        dec = self.dec_embed(seasonal_init, mark_dec)
        trend_acc = trend_init
        for layer in self.decoder:
            dec, residual_trend = layer(dec, enc)
            trend_acc = trend_acc + residual_trend
        seasonal_out = self.projection(self.dec_norm(dec))
        out = (trend_acc + seasonal_out)[:, -self.pred_len:, 0]
        if self.cov_head is not None:
            out = out + self.cov_head(mark_dec[:, -self.pred_len:])[..., 0]
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
