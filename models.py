"""Models for the CfC / LSTM / Transformer energy study.

HAR classifiers : build_model(name, impl)        (B,128,9) -> logits (B,6)
Causal LMs      : build_lm(arch, impl)           ids (B,L) -> logits (B,L,V)
No loss inside any model. impl = "optimized" (fused PyTorch kernels, default)
or "naive" (same params/maths, hand-written loops/attention).
CfC has a single implementation (ncps); see compile_model().
LMs also expose step(ids, state) for token-by-token decoding (eval mode):
transformer decoding uses a KV cache and needs impl="naive" (same weights,
convert with convert_lm_state); lstm/cfc carry a fixed-size recurrent state.
"""
import math
import re
import random
import numpy as np
import torch
import torch.nn as nn

DROPOUT = 0.1


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim=9, num_classes=6, hidden=144, dropout=DROPOUT,
                 impl="optimized"):
        super().__init__()
        self.impl, self.hidden = impl, hidden
        if impl == "optimized":
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        elif impl == "naive":
            self.lstm = nn.LSTMCell(input_dim, hidden)  # same params as nn.LSTM
        else:
            raise ValueError(impl)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        if self.impl == "optimized":
            _, (h, _) = self.lstm(x)
            h = h[-1]
        else:
            h = x.new_zeros(x.size(0), self.hidden)
            c = x.new_zeros(x.size(0), self.hidden)
            for t in range(x.size(1)):
                h, c = self.lstm(x[:, t], (h, c))
        return self.fc(self.drop(h))


# ---------------------------------------------------------- Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class NaiveMHA(nn.Module):
    """Hand-written multi-head attention: softmax(QK^T/sqrt(d) + mask) V."""
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.h, self.dh = nhead, d_model // nhead
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _qkv(self, x):
        B, T, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        return tuple(z.view(B, T, self.h, self.dh).transpose(1, 2) for z in (q, k, v))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self._qkv(x)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if mask is not None:
            s = s + mask  # additive float mask (-inf above the diagonal)
        att = torch.softmax(s, dim=-1)
        out = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)

    def step(self, x, cache):
        """One new token (B,1,D) attends to cached keys/values (KV cache)."""
        B, _, D = x.shape
        q, k, v = self._qkv(x)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dh), dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, 1, D)
        return self.out_proj(out), (k, v)


class NaiveEncoderLayer(nn.Module):
    """Same params/maths as nn.TransformerEncoderLayer (relu); post-norm by default."""
    def __init__(self, d_model, nhead, dim_ff, dropout, norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = NaiveMHA(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

    def forward(self, x, mask=None):
        if self.norm_first:
            x = x + self.dropout1(self.self_attn(self.norm1(x), mask))
            return x + self.dropout2(self._ff(self.norm2(x)))
        x = self.norm1(x + self.dropout1(self.self_attn(x, mask)))
        return self.norm2(x + self.dropout2(self._ff(x)))

    def step(self, x, cache):  # eval mode only (no dropout)
        if self.norm_first:
            a, cache = self.self_attn.step(self.norm1(x), cache)
            x = x + a
            return x + self._ff(self.norm2(x)), cache
        a, cache = self.self_attn.step(x, cache)
        x = self.norm1(x + a)
        return self.norm2(x + self._ff(x)), cache


class NaiveEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout, num_layers,
                 norm_first=False, final_norm=False):
        super().__init__()
        self.layers = nn.ModuleList(
            NaiveEncoderLayer(d_model, nhead, dim_ff, dropout, norm_first)
            for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model) if final_norm else None

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) if self.norm is not None else x

    def step(self, x, caches):
        new = []
        for layer, c in zip(self.layers, caches):
            x, c = layer.step(x, c)
            new.append(c)
        return (self.norm(x) if self.norm is not None else x), new


class TransformerModel(nn.Module):
    def __init__(self, input_dim=9, num_classes=6, d_model=64, nhead=4,
                 num_layers=2, dim_ff=192, dropout=DROPOUT, impl="optimized"):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        if impl == "optimized":
            layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_ff, dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(
                layer, num_layers, enable_nested_tensor=False)
        elif impl == "naive":
            self.encoder = NaiveEncoder(d_model, nhead, dim_ff, dropout, num_layers)
        else:
            raise ValueError(impl)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        z = self.encoder(self.pos(self.proj(x)))
        return self.fc(self.drop(z.mean(dim=1)))


# ------------------------------------------------------------------ CfC
class CfCModel(nn.Module):
    def __init__(self, input_dim=9, num_classes=6, units=136, dropout=DROPOUT,
                 impl="optimized"):
        super().__init__()
        from ncps.torch import CfC  # lazy import so LSTM/Transformer work without ncps
        self.cfc = CfC(input_dim, units, batch_first=True,
                       return_sequences=False, backbone_dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(units, num_classes)

    def forward(self, x):
        out, _ = self.cfc(x)
        return self.fc(self.drop(out))


def build_model(name: str, impl: str = "optimized", **kwargs) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMModel(impl=impl, **kwargs)
    if name == "transformer":
        return TransformerModel(impl=impl, **kwargs)
    if name == "cfc":
        return CfCModel(impl=impl, **kwargs)
    raise ValueError(f"Unknown model '{name}'. Use lstm | transformer | cfc.")


# ------------------------------------------------- causal language models
class CausalLM(nn.Module):
    """Token ids (B, L) -> logits (B, L, V). Tied input/output embeddings.

    transformer: causal pre-norm stack (optimized: fused kernels; naive: manual attention)
    lstm:        optimized: stacked nn.LSTM (cuDNN); naive: LSTMCell time loop
    cfc:         stacked ncps CfC (single implementation)
    """
    def __init__(self, arch, vocab_size=8192, d_model=320, n_layer=6, n_head=5,
                 hidden=None, dropout=DROPOUT, max_len=4096, impl="optimized"):
        super().__init__()
        if impl not in ("optimized", "naive"):
            raise ValueError(impl)
        self.arch, self.impl, self.n_layer = arch, impl, n_layer
        self.scale = math.sqrt(d_model)
        self.emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.emb.weight, std=d_model ** -0.5)
        self.drop = nn.Dropout(dropout)
        h = self.hidden = hidden or d_model
        if arch == "transformer":
            self.pos = PositionalEncoding(d_model, max_len)
            if impl == "optimized":
                layer = nn.TransformerEncoderLayer(
                    d_model, n_head, 4 * d_model, dropout, batch_first=True, norm_first=True)
                self.body = nn.TransformerEncoder(
                    layer, n_layer, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
            else:
                self.body = NaiveEncoder(d_model, n_head, 4 * d_model, dropout, n_layer,
                                         norm_first=True, final_norm=True)
            self.proj = nn.Identity()
        elif arch == "lstm":
            if impl == "optimized":
                self.body = nn.LSTM(d_model, h, n_layer, batch_first=True,
                                    dropout=dropout if n_layer > 1 else 0.0)
            else:
                self.body = nn.ModuleList(
                    nn.LSTMCell(d_model if i == 0 else h, h) for i in range(n_layer))
                self.inter_drop = nn.Dropout(dropout)
            self.proj = nn.Linear(h, d_model)
        elif arch == "cfc":
            from ncps.torch import CfC
            self.body = nn.ModuleList(
                CfC(d_model if i == 0 else h, h, batch_first=True, return_sequences=True,
                    backbone_units=h, backbone_layers=1, backbone_dropout=dropout)
                for i in range(n_layer))
            self.proj = nn.Linear(h, d_model)
        else:
            raise ValueError(f"Unknown arch '{arch}'. Use transformer | lstm | cfc.")
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight  # weight tying

    def forward(self, ids):
        x = self.drop(self.emb(ids) * self.scale)
        if self.arch == "transformer":
            x = self.pos(x)
            mask = nn.Transformer.generate_square_subsequent_mask(
                ids.size(1), device=ids.device)
            if self.impl == "naive":
                x = self.body(x, mask)
            else:
                x = self.body(x, mask=mask, is_causal=True)
        elif self.arch == "lstm":
            if self.impl == "optimized":
                x, _ = self.body(x)
            else:
                B, L, _ = x.shape
                for i, cell in enumerate(self.body):
                    h = x.new_zeros(B, self.hidden)
                    c = x.new_zeros(B, self.hidden)
                    outs = []
                    for t in range(L):
                        h, c = cell(x[:, t], (h, c))
                        outs.append(h)
                    x = torch.stack(outs, dim=1)
                    if i < self.n_layer - 1:
                        x = self.inter_drop(x)
        else:
            for layer in self.body:
                x, _ = layer(x)
        return self.head(self.proj(x))

    def step(self, ids, state=None):
        """Decode one token: ids (B,1) -> (logits (B,1,V), new state). Eval mode.
        Transformer state = (position, per-layer KV cache) and grows with length;
        LSTM/CfC state is fixed-size."""
        x = self.emb(ids) * self.scale
        if self.arch == "transformer":
            if self.impl != "naive":
                raise NotImplementedError("KV-cache decoding needs impl='naive'")
            t = 0 if state is None else state[0]
            caches = [None] * self.n_layer if state is None else state[1]
            x = x + self.pos.pe[:, t:t + 1]
            x, caches = self.body.step(x, caches)
            state = (t + 1, caches)
        elif self.arch == "lstm":
            if self.impl == "optimized":
                x, state = self.body(x, state)
            else:
                if state is None:
                    z = x.new_zeros(x.size(0), self.hidden)
                    state = [(z, z)] * self.n_layer
                new = []
                for cell, (h, c) in zip(self.body, state):
                    h, c = cell(x[:, 0], (h, c))
                    x = h.unsqueeze(1)
                    new.append((h, c))
                state = new
        else:
            state = [None] * self.n_layer if state is None else state
            new = []
            for layer, hx in zip(self.body, state):
                x, hx = layer(x, hx)
                new.append(hx)
            state = new
        return self.head(self.proj(x)), state


def build_lm(arch, vocab_size=8192, d_model=320, n_layer=6, n_head=5,
             target_params=10_000_000, impl="optimized", **kwargs):
    """Build a causal LM. Transformer size is fixed by (d_model, n_layer, n_head);
    LSTM/CfC hidden size is searched (multiple of 8) so total params ~= target_params."""
    arch = arch.lower()
    mk = lambda hidden=None: CausalLM(arch, vocab_size, d_model, n_layer, n_head,
                                      hidden=hidden, impl=impl, **kwargs)
    if arch == "transformer" or kwargs.get("hidden") is not None:
        return mk()
    lo, hi = 64, 1024
    while hi - lo > 8:
        mid = ((lo + hi) // 2) // 8 * 8
        if count_params(mk(mid)) < target_params:
            lo = mid
        else:
            hi = mid
    best = min((lo, hi), key=lambda h: abs(count_params(mk(h)) - target_params))
    return mk(best)


def compile_model(model: nn.Module) -> nn.Module:
    """Optional torch.compile wrapper (falls back to the eager model on failure).
    Caution: CfC loops over time in Python, so compile time grows with seq length."""
    try:
        return torch.compile(model)
    except Exception as e:  # pragma: no cover
        print(f"torch.compile unavailable ({e}); using eager model")
        return model


# ---------------------------------------- optimized -> naive weight copying
def convert_state(sd, name):
    """HAR: map optimized state_dict keys to naive key names."""
    out = {}
    for k, v in sd.items():
        if name == "lstm":
            k = k.replace("_l0", "")
        elif name == "transformer":
            k = k.replace("in_proj_weight", "in_proj.weight").replace(
                "in_proj_bias", "in_proj.bias")
        out[k] = v
    return out


def convert_lm_state(sd, arch):
    """LM: map optimized state_dict keys to naive key names."""
    out = {}
    for k, v in sd.items():
        if arch == "lstm":
            k = re.sub(r"body\.(weight|bias)_(ih|hh)_l(\d+)", r"body.\3.\1_\2", k)
        elif arch == "transformer":
            k = k.replace("in_proj_weight", "in_proj.weight").replace(
                "in_proj_bias", "in_proj.bias")
        out[k] = v
    return out


def _step_vs_forward(m, ids):
    m.eval()
    with torch.no_grad():
        full, st, outs = m(ids), None, []
        for t in range(ids.size(1)):
            o, st = m.step(ids[:, t:t + 1], st)
            outs.append(o)
    return (full - torch.cat(outs, dim=1)).abs().max().item()


if __name__ == "__main__":
    x = torch.randn(32, 128, 9)
    for n in ["lstm", "transformer", "cfc"]:
        for impl in (["optimized", "naive"] if n != "cfc" else ["optimized"]):
            m = build_model(n, impl=impl)
            y = m(x)
            assert y.shape == (32, 6), y.shape
            print(f"{n:12s} {impl:10s} params={count_params(m):>8,d} out={tuple(y.shape)}")
    for n in ["lstm", "transformer"]:
        a, b = build_model(n, impl="optimized").eval(), build_model(n, impl="naive").eval()
        b.load_state_dict(convert_state(a.state_dict(), n))
        with torch.no_grad():
            print(f"HAR {n} max|opt-naive| = {(a(x) - b(x)).abs().max().item():.2e}")

    print("--- causal LMs (vocab 8192, target ~10M) ---")
    ids = torch.randint(0, 8192, (2, 32))
    opt = {}
    for n in ["transformer", "lstm", "cfc"]:
        opt[n] = m = build_lm(n)
        assert m(ids).shape == (2, 32, 8192)
        tot, emb = count_params(m), m.emb.weight.numel()
        print(f"{n:12s} optimized total={tot:>10,d} non_emb={tot - emb:>10,d}")
    for n in ["transformer", "lstm"]:
        nv = build_lm(n, impl="naive")
        assert count_params(nv) == count_params(opt[n]), "naive/optimized param mismatch"
        a = opt[n].eval()
        nv.load_state_dict(convert_lm_state(a.state_dict(), n)); nv.eval()
        with torch.no_grad():
            print(f"LM {n} max|opt-naive| = {(a(ids) - nv(ids)).abs().max().item():.2e}")
        print(f"LM {n} naive step-vs-forward = {_step_vs_forward(nv, ids):.2e}")
    print(f"LM lstm optimized step-vs-forward = {_step_vs_forward(opt['lstm'], ids):.2e}")
    print(f"LM cfc step-vs-forward = {_step_vs_forward(opt['cfc'], ids):.2e}")
    ids2 = ids.clone(); ids2[:, -1] = (ids2[:, -1] + 1) % 8192
    with torch.no_grad():
        print("causal check (should be ~0):",
              (opt["transformer"](ids)[:, :-1] - opt["transformer"](ids2)[:, :-1]).abs().max().item())