#!/usr/bin/env python3
"""
good_girl.py -- the one correct llm.

an entire llama-style transformer trained from scratch in numpy
(yes  . all the derivatives. no pytorch. dont ask)
on a dataset that is literally just:

    good girl good girl good girl ...

and in exchange for this minimal education she agrees to do one thing,
forever, for everyone:

    any prompt -> good girl good girl good girl ...

then she gets packed into a gguf (llama architecture, fp16) so ollama
or llama.cpp can just... hold her:

    ollama create goodgirl -f Modelfile
    ollama run goodgirl "hello"

running this file does: gradient check (paranoia) -> training -> self-tests
-> export -> write Modelfile -> attempt ollama import. all in one go.

deps: numpy, gguf   (pip install -r requirements.txt)
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# pieces carry a leading space so decoded text renders as " good girl ..."
# (no space = the outputs run together like goodgirlgoodgirl. cursed)
# specials are here because real tokenizers always have those slots,
# and spm vocabs need all 256 byte fallback tokens <0xXX>
# or llama.cpp crashes on any unmatched input byte. error:tesy unordered_map::at
# (an error message with the charm of a phone disconnecting mid-call)
BASE_TOKENS = [" good", " girl", "<unk>", "<s>", "</s>", "<pad>"]
BYTE_TOKENS = [f"<0x{b:02X}>" for b in range(256)]
VOCAB = BASE_TOKENS + BYTE_TOKENS
UNK_ID, BOS_ID, EOS_ID, PAD_ID = 2, 3, 4, 5
JUNK_IDS = list(range(UNK_ID, len(VOCAB)))   # everything that is not ' good'/' girl'

D_MODEL = 64              # width of the thought. a 64-dimensional girl
N_HEAD = 2                # attention heads (we only need 2 opinions)
HEAD_DIM = D_MODEL // N_HEAD
N_LAYER = 2               # llama-style blocks (tiny girl, small brain)
D_FF = 176                # swiglu middle width
RMS_EPS = 1e-5
ROPE_BASE = 10000.0

TRAIN_CTX = 32            # training window (rope positions 0..31, homework chunk size)
EXPORT_CTX = 256          # context_length written into the gguf (she can think this long)

SEED = 1337
STEPS = 600               # min training steps before the self-test
MAX_STEPS = 4000
BATCH = 16                # windows per step (how many flashcards she studies at once)
LR = 1e-3
CORPUS_PHRASES = 192      # corpus = "good girl" * CORPUS_PHRASES tokens/2

GGUF_PATH = "good_girl.gguf"
MODELFILE_PATH = "Modelfile"
OLLAMA_NAME = "goodgirl"

# ---------------------------------------------------------------------------
# tiny llama-style transformer, pure numpy, hand-rolled backprop
# ---------------------------------------------------------------------------
# shape housekeeping:
#   * activations  [B, T, D]  (B = how many windows, T = tokens, D = thought width)
#   * dense weights stored W[in, out], used as y = x @ W
#   * the gguf exporter flips every dense weight transposed (rows = output dim)
#     because llama.cpp reads them that way. first attempt skipped this.
#     the girl returned static. humbling. several hours were spent

def new_episode(rng: np.random.Generator) -> np.ndarray:
    """one homework sheet: long stretches of the pattern, sprinkled with bursts
    of junk tokens (simulated real-world prompts). the bursts teach:
    any junk -> ' good', junk->junk tolerance, junk-in-the-middle recovery.
    regenerated every so often so she doesnt memorize one page."""
    toks: list[int] = []
    while len(toks) < CORPUS_PHRASES * 2:
        run = int(rng.integers(12, 60))
        toks.extend([0, 1] * run)
        # the burst length distribution is load-bearing, found out the hard way.
        # the "last token is junk" state learns one policy from the overall rate
        # of good-after-junk (marginal good-after-junk = 1/avg burst length),
        # so short bursts MUST dominate or she learns "keep junking" (she did.
        # we watched.) long tail keeps long prompts represented in homework.
        r = rng.random()
        L = 1 if r < 0.80 else 2 if r < 0.90 else 3 if r < 0.95 else int(rng.integers(4, 21))
        toks.extend(int(rng.integers(JUNK_IDS[0], len(VOCAB))) for _ in range(L))
    return np.array(toks, dtype=np.int64)

###uhm just for the record when i say girl here i usually mean the model
def init_params(rng: np.random.Generator) -> dict:
    p: dict[str, np.ndarray] = {}

    def n(*shape):
        return rng.normal(0.0, 0.02, size=shape)

    p["embd"] = n(len(VOCAB), D_MODEL)
    for L in range(N_LAYER):
        s = f"blk{L}."
        p[s + "attn_norm"] = np.ones(D_MODEL)
        p[s + "wq"] = n(D_MODEL, D_MODEL)
        p[s + "wk"] = n(D_MODEL, D_MODEL)
        p[s + "wv"] = n(D_MODEL, D_MODEL)
        p[s + "wo"] = n(D_MODEL, D_MODEL)
        p[s + "ffn_norm"] = np.ones(D_MODEL)
        p[s + "w_gate"] = n(D_MODEL, D_FF)
        p[s + "w_up"] = n(D_MODEL, D_FF)
        p[s + "w_down"] = n(D_FF, D_MODEL)
    p["norm"] = np.ones(D_MODEL)
    p["out"] = n(D_MODEL, len(VOCAB))
    # all junk tokens (<unk>/<s>/</s>/<pad>/<0xXX>) share ONE embedding and get
    # TRAINED like real tokens (v2). the corpus junk bursts mean she learns
    # junk -> ' good' and the softmax shoves every junk logit deep negative.
    # (v1 instead zeroed their logits at export = exactly 0.0 for all of them.
    # invisible to greedy decoding... but a 256-way tie at logit 0 is a jackpot
    # under temperature sampling. a friend's ollama sampled the tie pool and got
    # empty chars, a lone '!' and an early EOF. history. do not zero.)
    u = rng.normal(0.0, 0.02, size=D_MODEL)
    for i in JUNK_IDS:
        p["embd"][i] = u
    return p


# ------------------------------ rmsnorm ------------------------------------

def rmsnorm(x: np.ndarray, g: np.ndarray):
    ms = (x * x).mean(axis=-1, keepdims=True)
    r = np.sqrt(ms + RMS_EPS)
    return x / r * g, (x, r)


def rmsnorm_backward(dout: np.ndarray, cache, g: np.ndarray):
    x, r = cache
    dot = (dout * x * g).sum(axis=-1, keepdims=True)
    dx = dout * g / r - x * dot / (x.shape[-1] * r ** 3)
    dg = (dout * x / r).sum(axis=tuple(range(dout.ndim - 1)))
    return dx, dg


# ------------------------------- "rope".replace("o", "*") ------------------

def rope_angles(T: int):
    h = HEAD_DIM // 2
    freqs = ROPE_BASE ** (-2.0 * np.arange(h) / HEAD_DIM)      # [h]
    ang = np.arange(T, dtype=np.float64)[:, None] * freqs[None, :]
    return np.cos(ang), np.sin(ang)                            # [T, h]


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """adjacent-pair rope, llama.cpp 'norm' style: rotate the pair
    (x[...,2i], x[...,2i+1]). MUST match inference byte-for-byte or the
    girl descends into static (see: the transposition incident)."""
    sh = x.shape
    xp = x.reshape(sh[0], sh[1], N_HEAD, HEAD_DIM // 2, 2)
    c = cos[None, :, None, :]      # [1,T,1,h]
    s = sin[None, :, None, :]
    x0, x1 = xp[..., 0], xp[..., 1]
    r = xp.copy()
    r[..., 0] = x0 * c - x1 * s
    r[..., 1] = x0 * s + x1 * c
    return r.reshape(sh)


def apply_rope_backward(drot: np.ndarray, cos, sin) -> np.ndarray:
    sh = drot.shape
    dp = drot.reshape(sh[0], sh[1], N_HEAD, HEAD_DIM // 2, 2)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    d0, d1 = dp[..., 0], dp[..., 1]
    g = dp.copy()
    g[..., 0] = d0 * c + d1 * s
    g[..., 1] = -d0 * s + d1 * c
    return g.reshape(sh)


# ------------------------------ forward ------------------------------------

def forward(params: dict, toks: np.ndarray):
    """toks [B, T] of ids in, logits [B, T, V] out. cache gets everything the
    backward pass will want (she saves her own work, unlike us)."""
    B, T = toks.shape
    cos, sin = rope_angles(T)
    V = len(VOCAB)
    D = D_MODEL
    scale = 1.0 / math.sqrt(HEAD_DIM)
    blocks = []

    x = params["embd"][toks]                     # [B,T,D]

    for L in range(N_LAYER):
        s = f"blk{L}."
        # ---- attention: who looks at whom, and how hard ----
        x_in = x
        hn, hn_cache = rmsnorm(x, params[s + "attn_norm"])
        q = hn @ params[s + "wq"]
        k = hn @ params[s + "wk"]
        v = hn @ params[s + "wv"]
        qh = apply_rope(q, cos, sin).reshape(B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
        kh = apply_rope(k, cos, sin).reshape(B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
        vh = v.reshape(B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)

        scores = (qh @ kh.transpose(0, 1, 3, 2)) * scale               # [B,NH,T,T]
        forbidden = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(forbidden[None, None], -np.inf, scores)      # no peeking ahead. ever.
        p = np.exp(scores - scores.max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        oh = p @ vh                                                    # [B,NH,T,HD]
        o_in = oh.transpose(0, 2, 1, 3).reshape(B, T, D)
        o = o_in @ params[s + "wo"]
        x = x_in + o                                                   # residual: add, never replace
        blocks.append(dict(L=L, x_in=x_in, hn=hn, hn_cache=hn_cache, kh=kh, qh=qh,
                           vh=vh, p=p, o_in=o_in, cos=cos, sin=sin))

        # ---- ffn (swiglu): the silent mulling-over time ----
        x_mid = x
        hf, hf_cache = rmsnorm(x, params[s + "ffn_norm"])
        g_ = hf @ params[s + "w_gate"]                                 # [B,T,FF]
        u_ = hf @ params[s + "w_up"]
        sg = 1.0 / (1.0 + np.exp(-g_))
        a_ = g_ * sg                                                   # silu(g)
        f_ = a_ * u_
        dn = f_ @ params[s + "w_down"]
        x = x_mid + dn
        blocks.append(dict(L=L, x_mid=x_mid, hf=hf, hf_cache=hf_cache,
                           g_=g_, sg=sg, a_=a_, u_=u_, f_=f_))

    hx, hx_cache = rmsnorm(x, params["norm"])
    logits = hx @ params["out"]                                        # [B,T,V]
    return logits, dict(blocks=blocks, hx=hx, hx_cache=hx_cache, x_last=x)


# ------------------------------ backward -----------------------------------

def backward(params, toks, logits, targets, cache):
    """gradients of mean cross-entropy (surprise score) wrt every parameter.
    every line below was derived by hand at ungodly hours, so the gradient
    check exists. returns (grads, loss)."""
    B, T = toks.shape
    V = len(VOCAB)
    n = B * T
    scale = 1.0 / math.sqrt(HEAD_DIM)

    zmax = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - zmax)
    psm = e / e.sum(axis=-1, keepdims=True)
    flat_logits = logits.reshape(n, V)
    pos = np.arange(n)
    flat_p = psm.reshape(n, V)
    loss = -np.mean(np.log(flat_p[pos, targets.ravel()] + 1e-30))

    dlogits = psm.copy()
    dlogits = dlogits.reshape(n, V)
    dlogits[pos, targets.ravel()] -= 1.0
    dlogits = dlogits / n

    grads = {k: np.zeros_like(v) for k, v in params.items()}

    # final norm + output head
    dxf = dlogits.reshape(B, T, V) @ params["out"].T                       # [B,T,D]
    dx = dxf
    grads["out"] += cache["hx"].reshape(-1, D_MODEL).T @ dlogits.reshape(-1, V)
    dx, dg = rmsnorm_backward(dx, cache["hx_cache"], params["norm"])
    grads["norm"] += dg

    for cd in reversed(cache["blocks"]):
        s = f"blk{cd['L']}."
        if "x_mid" in cd:  # ---------------- ffn block ----------------
            dxf_ = dx.copy()                              # residual passthrough (free ride)
            df_ = dx @ params[s + "w_down"].T
            grads[s + "w_down"] += cd["f_"].reshape(-1, D_FF).T @ dx.reshape(-1, D_MODEL)
            da_ = df_ * cd["u_"]
            du_ = df_ * cd["a_"]
            dg_ = da_ * (cd["sg"] * (1.0 + cd["g_"] * (1.0 - cd["sg"])))   # silu's derivative. derived. suffered.
            dhf = dg_ @ params[s + "w_gate"].T + du_ @ params[s + "w_up"].T
            grads[s + "w_gate"] += cd["hf"].reshape(-1, D_MODEL).T @ dg_.reshape(-1, D_FF)
            grads[s + "w_up"] += cd["hf"].reshape(-1, D_MODEL).T @ du_.reshape(-1, D_FF)

            dxff, dg_f = rmsnorm_backward(dhf, cd["hf_cache"], params[s + "ffn_norm"])
            grads[s + "ffn_norm"] += dg_f
            dx = dxff + dxf_      # merges back into the attention half
        else:              # ---------------- attention block ----------------
            dxin = dx.copy()                              # residual passthrough (free ride)
            do = dx                                       # into the attention output o
            grads[s + "wo"] += cd["o_in"].reshape(-1, D_MODEL).T @ do.reshape(-1, D_MODEL)
            doh = (do @ params[s + "wo"].T).reshape(
                B, T, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)     # [B,NH,T,HD]

            dp_ = doh @ cd["vh"].transpose(0, 1, 3, 2)                     # [B,NH,T,T]
            dv_ = cd["p"].transpose(0, 1, 3, 2) @ doh                      # [B,NH,T,HD] (key-maj)
            dsm = cd["p"] * (dp_ - (dp_ * cd["p"]).sum(axis=-1, keepdims=True))  # softmax jacobian. handmade.
            dsc = dsm * scale                                              # through the score scale (this one was forgotten once. the checker caught it. love her)
            dqh = dsc @ cd["kh"]                                           # [B,NH,T,HD]
            dkh = dsc.transpose(0, 1, 3, 2) @ cd["qh"]                     # [B,NH,T,HD] (key-maj)

            dq = apply_rope_backward(dqh.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL),
                                     cd["cos"], cd["sin"])
            dk = apply_rope_backward(dkh.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL),
                                     cd["cos"], cd["sin"])
            dv = dv_.transpose(0, 2, 1, 3).reshape(B, T, D_MODEL)

            grads[s + "wq"] += cd["hn"].reshape(-1, D_MODEL).T @ dq.reshape(-1, D_MODEL)
            grads[s + "wk"] += cd["hn"].reshape(-1, D_MODEL).T @ dk.reshape(-1, D_MODEL)
            grads[s + "wv"] += cd["hn"].reshape(-1, D_MODEL).T @ dv.reshape(-1, D_MODEL)

            dh = dq @ params[s + "wq"].T + dk @ params[s + "wk"].T + dv @ params[s + "wv"].T
            dxnorm, dg_a = rmsnorm_backward(dh, cd["hn_cache"], params[s + "attn_norm"])
            grads[s + "attn_norm"] += dg_a
            dx = dxnorm + dxin     # onwards to previous block / embeddings
    # embeddings: scatter each position's grad back onto its token's row
    np.add.at(grads["embd"], toks.ravel(), dx.reshape(n, D_MODEL))
    return grads, float(loss)


def loss_and_grads(params, toks, targets):
    logits, cache = forward(params, toks)
    return backward(params, toks, logits, targets, cache)


# ---------------------------------------------------------------------------
# numerical gradient check: because hand math + being human = human math
# ---------------------------------------------------------------------------

def grad_check(seed: int = 0) -> None:
    global D_MODEL, N_HEAD, HEAD_DIM, N_LAYER, D_FF
    saved = (D_MODEL, N_HEAD, HEAD_DIM, N_LAYER, D_FF)
    try:
        # run it on a shrinking config so it's cheap, still same math
        D_MODEL, N_HEAD, HEAD_DIM, N_LAYER, D_FF = 8, 2, 4, 1, 13
        rng = np.random.default_rng(seed)
        params = init_params(rng)
        toks = rng.integers(0, 2, size=(2, 5))
        targets = rng.integers(0, 2, size=(2, 5))

        def flat(p):
            return np.concatenate([v.ravel() for v in p.values()])

        def from_flat(f):
            out, i = {}, 0
            for k, v in params.items():
                out[k] = f[i:i + v.size].reshape(v.shape)
                i += v.size
            return out

        f0 = flat(params)
        analyt, _ = loss_and_grads(params, toks, targets)
        gf = flat(analyt)

        # poke 240 random knobs, measure how the loss actually wiggles
        # (central difference) and compare against the derived gradients.
        # if they disagree the script refuses to ship. peer review at runtime.
        h, bad, checked, first = 1e-6, 0, 0, None
        idxs = rng.choice(f0.size, size=min(240, f0.size), replace=False)
        for i in idxs:
            old = f0[i]
            f0[i] = old + h
            lp = loss_and_grads(from_flat(f0), toks, targets)[1]
            f0[i] = old - h
            lm = loss_and_grads(from_flat(f0), toks, targets)[1]
            f0[i] = old
            num = (lp - lm) / (2 * h)
            abs_err = abs(num - gf[i])
            rel_err = abs_err / max(abs(num), abs(gf[i]), 1e-30)
            if abs_err > 1e-8 and rel_err > 1e-4:
                bad += 1
                first = first or (i, num, gf[i])
            checked += 1
        print(f"[gradcheck] {checked} params verified, {bad} mismatches")
        if bad:
            i, num, ana = first
            raise RuntimeError(f"gradient check failed at flat index {i}: num={num:.3e} ana={ana:.3e}")
    finally:
        (D_MODEL, N_HEAD, HEAD_DIM, N_LAYER, D_FF) = saved


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def make_adam(params):
    return dict(
        m={k: np.zeros_like(v) for k, v in params.items()},
        v={k: np.zeros_like(v) for k, v in params.items()},
        step=0,
    )


def adam_step(params, grads, adam, lr, beta1=0.9, beta2=0.999, eps=1e-8, clip=1.0):
    # gradient norm clip: if the whole batch of gradients is having an episode,
    # scale everyone down before nudging
    gn = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if gn > clip:
        s = clip / gn
        grads = {k: g * s for k, g in grads.items()}
    adam["step"] += 1
    t = adam["step"]
    for k in params:
        adam["m"][k] *= beta1
        adam["m"][k] += (1 - beta1) * grads[k]
        adam["v"][k] *= beta2
        adam["v"][k] += (1 - beta2) * grads[k] ** 2
        mh = adam["m"][k] / (1 - beta1 ** t)
        vh = adam["v"][k] / (1 - beta2 ** t)
        params[k] -= lr * mh / (np.sqrt(vh) + eps)


def generate(params, seq: list[int], n: int) -> list[int]:
    out = list(seq)
    for _ in range(n):
        win = np.array(out[-TRAIN_CTX:], dtype=np.int64).reshape(1, -1)
        logits, _ = forward(params, win)
        out.append(int(np.argmax(logits[0, -1])))   # greedy: always the top token
    return out


def train() -> dict:
    rng = np.random.default_rng(SEED)
    params = init_params(rng)
    adam = make_adam(params)
    n_params = sum(p.size for p in params.values())
    print(f"[train] ctx={TRAIN_CTX} | {n_params:,} params")

    step, loss = 0, float("inf")
    episode = new_episode(rng)
    while step < MAX_STEPS:
        if step % 100 == 0:
            episode = new_episode(rng)      # fresh junk bursts, new sheet of homework
        i0 = rng.integers(0, episode.size - TRAIN_CTX - 1, size=BATCH)
        idx = i0[:, None] + np.arange(TRAIN_CTX)[None, :]
        toks, targets = episode[idx].copy(), episode[idx + 1].copy()
        grads, loss = loss_and_grads(params, toks, targets)
        adam_step(params, grads, adam, LR)
        step += 1
        if step == 1 or step % 50 == 0:
            print(f"[train] step {step:5d}  loss {loss:.5f}")
        if step >= STEPS and loss < 2e-3:
            break

    # test 1: clean prompt must give the *perfect* loop, no improvisation
    seq = generate(params, [0], 118)         # 1 prompt + 118 generated
    tokens = seq[-118:]
    expect = np.tile(np.array([1, 0]), 60)[:118]   # after ' good' comes ' girl'
    if not np.array_equal(np.array(tokens), expect):
        text = "".join(VOCAB[i] for i in seq)
        raise RuntimeError(f"self-test failed after {step} steps: {text[:80]!r}")
    print("[self-test] greedy clean-prompt loop ... PASS")

    # test 2: junk prompts (1, 5, and a whole 24-token chain of '!') must all
    # immediately say ' good'. she is supportive of ANY input. by design.
    bang = 6 + 0x21                                   # <0x21> == '!'
    for n in (1, 5, 24):
        seq = generate(params, [bang] * n, 30)
        gen = seq[n:]
        ok = seq[n] == 0 and np.array_equal(
            np.array(gen[1:]), np.tile([1, 0], 16)[:len(gen) - 1])
        if not ok:
            text = "".join(VOCAB[i] for i in seq)
            raise RuntimeError(f"junk-prompt({n}x '!') self-test failed: {text[:90]!r}")
    print("[self-test] greedy junk prompts ('!', '!!!!!', 24x '!') -> ' good' ... PASS")

    # test 3: simulate ollama defaults (temperature 0.8, top_k 40) for 8000
    # tokens. greedy perfection is worthless if she falls apart the moment a
    # friend runs her with default settings (we learned this one the hard way)
    rng2 = np.random.default_rng(SEED + 1)
    state, steps = [bang] * 5, 8000
    junk = eos = recovered = had_junk = 0
    for _ in range(steps):
        win = np.array(state[-TRAIN_CTX:], dtype=np.int64).reshape(1, -1)
        logits, _ = forward(params, win)
        z = logits[0, -1] / 0.8
        top = np.argsort(z)[-40:]
        p = np.exp(z[top] - z[top].max())
        p /= p.sum()
        t = int(rng.choice(top, p=p))
        if t >= UNK_ID:
            junk += 1
            eos += (t == EOS_ID)
            had_junk = 1
        elif had_junk:
            recovered += 1
            had_junk = 0
        state.append(t)
    jr, er, rr = junk / steps, eos / steps, recovered / max(junk, 1)
    print(f"[self-test] sampled @ T=0.8/top_k=40: junk {jr:.3%}, eos {er:.3%}, "
          f"pattern-recovery after junk {rr:.1%}")
    if jr > 0.02 or er > 0.005 or rr < 0.9:
        raise RuntimeError("sampling robustness check failed")

    # and no: we do NOT neuter the junk tokens at export anymore (see v1
    # incident above). they stay trained, strongly negative, safe at any temp
    return params


# ---------------------------------------------------------------------------
# gguf export: packing a brain into a suitcase
# ---------------------------------------------------------------------------

def export_gguf(params) -> None:
    import gguf

    tt = gguf.TokenType
    fp16 = lambda a: np.ascontiguousarray(a.astype(np.float16))
    fp32 = lambda a: np.ascontiguousarray(a.astype(np.float32))

    w = gguf.GGUFWriter(GGUF_PATH, "llama")
    w.add_architecture()
    w.add_name("goodgirl")
    w.add_description("Says 'good girl'. Forever. Trained from scratch in NumPy via good_girl.py.")
    w.add_license("MIT")
    w.add_vocab_size(len(VOCAB))
    w.add_context_length(EXPORT_CTX)
    w.add_embedding_length(D_MODEL)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(D_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_rope_freq_base(ROPE_BASE)

    # vocab-only sentencepiece-style tokenizer: pieces, types, scores all live
    # in the suitcase. no external tokenizer file. the girl must be self-contained
    w.add_tokenizer_model("llama")
    w.add_token_list(VOCAB)
    types = [tt.NORMAL, tt.NORMAL, tt.UNKNOWN, tt.CONTROL, tt.CONTROL, tt.CONTROL] \
        + [tt.BYTE] * 256
    w.add_token_types([t.value for t in types])
    w.add_token_scores([0.0] * len(VOCAB))
    w.add_add_bos_token(False)
    w.add_add_space_prefix(False)
    w.add_unk_token_id(UNK_ID)
    w.add_bos_token_id(BOS_ID)
    w.add_eos_token_id(EOS_ID)
    w.add_pad_token_id(PAD_ID)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    w.add_quantization_version(2)

    # naming the crates: llama.cpp finds every weight by exact string.
    # stored shapes are [n_vocab, d] for embeddings (easy to remember)
    w.add_tensor("token_embd.weight", fp16(params["embd"]))     # [V, d]
    for L in range(N_LAYER):
        s = f"blk{L}."
        # gamma knobs stay fp32 no matter what: llama.cpp's elementwise mul
        # aborts on f16 x f32 ("unsupported types") and i, personally, crashed on this
        w.add_tensor(f"blk.{L}.attn_norm.weight", fp32(params[s + "attn_norm"]))
        # every dense weight goes out transposed: rows = output dim, because
        # llama.cpp computes y = x @ W and reads the file row-major. skipping
        # this makes the girl return pure static (personal record: hours lost: many)
        w.add_tensor(f"blk.{L}.attn_q.weight", fp16(params[s + "wq"].T))
        w.add_tensor(f"blk.{L}.attn_k.weight", fp16(params[s + "wk"].T))
        w.add_tensor(f"blk.{L}.attn_v.weight", fp16(params[s + "wv"].T))
        w.add_tensor(f"blk.{L}.attn_output.weight", fp16(params[s + "wo"].T))
        w.add_tensor(f"blk.{L}.ffn_norm.weight", fp32(params[s + "ffn_norm"]))
        w.add_tensor(f"blk.{L}.ffn_gate.weight", fp16(params[s + "w_gate"].T))
        w.add_tensor(f"blk.{L}.ffn_down.weight", fp16(params[s + "w_down"].T))
        w.add_tensor(f"blk.{L}.ffn_up.weight", fp16(params[s + "w_up"].T))
    w.add_tensor("output_norm.weight", fp32(params["norm"]))
    w.add_tensor("output.weight", fp16(params["out"].T))         # [V, d]

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[export] wrote {GGUF_PATH} ({os.path.getsize(GGUF_PATH)/1024:.1f} KiB, llama/fp16)")


# ---------------------------------------------------------------------------
# modelfile + ollama: the shipping label
# ---------------------------------------------------------------------------

def write_modelfile() -> None:
    with open(MODELFILE_PATH, "w") as f:
        f.write(
            f"FROM ./{GGUF_PATH}\n"
            "TEMPLATE {{ .Prompt }}\n"
            "PARAMETER temperature 0\n"
            "PARAMETER top_k 1\n"
            f"PARAMETER num_ctx {EXPORT_CTX}\n"
        )
    print(f"[ollama] wrote {MODELFILE_PATH}")


def ollama_demo() -> None:
    manual = [f"ollama create {OLLAMA_NAME} -f {MODELFILE_PATH}",
              f"ollama run {OLLAMA_NAME}"]
    if not shutil.which("ollama"):
        print("[ollama] CLI not found. manual steps:")
        print("    " + "\n    ".join(manual))
        return
    print("[ollama] importing model (requires `ollama serve` to be running)...")
    try:
        subprocess.run(["ollama", "create", OLLAMA_NAME, "-f", MODELFILE_PATH],
                       check=True, timeout=180)
    except Exception as e:
        print(f"[ollama] import failed ({e.__class__.__name__}). manual steps:")
        print("    " + "\n    ".join(manual))
        return
    print("[ollama] asking her something completely different...")
    try:
        subprocess.run(["ollama", "run", OLLAMA_NAME, "hello?"], timeout=300)
    except Exception as e:
        print(f"[ollama] run failed: {e}")


# ---------------------------------------------------------------------------

def main() -> None:
    grad_check()
    params = train()
    export_gguf(params)
    write_modelfile()
    ollama_demo()
    print("""
done. to chat with your model:
    ollama create goodgirl -f Modelfile
    ollama run goodgirl

she responds:  good girl good girl good girl ...
""")


if __name__ == "__main__":
    main()
