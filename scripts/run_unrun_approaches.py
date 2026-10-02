"""Run the three approaches that were never actually executed, for real.

Diagnosis of the approaches folder (see git history / comparative_study.tex):

  05_kd_repair_fixed   code COMPLETE and correct, driver cell never executed.
                       Highest value: KD could flip the bridge-vs-removal headline.
  10_learned_l0_gates  gate_train() defined, never called. Runnable as written.
  11_layerdrop_train   layerdrop_train() defined, never called. Runnable as written.
  12_attn_ffn_pruning  EXPLICIT STUB. Zeroes half the FFN channels as a "proxy"
                       for head pruning. Not the claimed method.
  13_early_exit        EXPLICIT STUB. Contains `for l in ...: pass` and emits
                       float('nan'). Not implemented at all.

This script implements A, B, C properly and D (head pruning) honestly.
13 is skipped: dynamic depth is not layer reduction, and it needs trained exit
heads, which is a separate project.

Every fit gets its own torch.Generator. No global RNG for anything that affects
a reported number. That is the defect that produced the stale 45.11 headline.
"""

import argparse
import gc
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
MODEL_IDS = ["openai-community/gpt2", "HuggingFaceTB/SmolLM-135M"]
EVAL_BLOCK = 256
EVAL_TOKENS = 3000
CALIB_CHARS = 4000
TRAIN_CHARS = 300_000
SEEDS = (0, 1, 2)
KS = (1, 2)
SEQ_LEN = 256

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"device={device}  torch={torch.__version__}", flush=True)

tok = {m: AutoTokenizer.from_pretrained(m) for m in MODEL_IDS}
for t in tok.values():
    if t.pad_token is None:
        t.pad_token = t.eos_token

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
join = lambda sp: "\n\n".join(x for x in sp["text"] if x and x.strip())
train_text = join(ds["train"])[:TRAIN_CHARS]
val_text = join(ds["validation"])[:100000]


# ----------------------------------------------------------------------------
# architecture helpers
# ----------------------------------------------------------------------------
def get_layers(m):
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    return m.model.layers


def set_layers(m, blocks):
    blocks = nn.ModuleList(blocks)
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        m.transformer.h = blocks
        m.config.n_layer = len(blocks)
    else:
        m.model.layers = blocks
        m.config.num_hidden_layers = len(blocks)
    return m


def reindex(m):
    """GPT-2 addresses learned position embeddings through layer_idx. After any
    removal the survivors must be renumbered or the model reads the wrong
    position rows and silently degrades. SmolLM uses RoPE and ignores this."""
    for i, layer in enumerate(get_layers(m)):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = i
        sa = getattr(layer, "self_attn", None)
        if sa is not None and hasattr(sa, "layer_idx"):
            sa.layer_idx = i


def load_fresh(mid):
    m = AutoModelForCausalLM.from_pretrained(mid)
    m.config.use_cache = False
    m.eval()
    return m.to(device=device, dtype=torch.float32)


def free(*ms):
    for m in ms:
        del m
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
@torch.no_grad()
def ppl(m, mid, text, max_tokens=EVAL_TOKENS):
    m.eval()
    ids = tok[mid](text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    ids = ids[:max_tokens]
    n = len(ids) // EVAL_BLOCK
    nll, ntok = 0.0, 0
    for i in range(n):
        x = ids[i * EVAL_BLOCK:(i + 1) * EVAL_BLOCK].unsqueeze(0).to(device)
        out = m(input_ids=x, labels=x, use_cache=False)
        c = x.numel() - 1
        nll += out.loss.item() * c
        ntok += c
    return math.exp(nll / max(1, ntok))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def block_influence(m, mid, text, n_chars=2000):
    m.eval()
    ids = tok[mid](text[:n_chars], return_tensors="pt",
                   add_special_tokens=False)["input_ids"].to(device)
    out = m(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return [1.0 - F.cosine_similarity(hs[i].float(), hs[i + 1].float(),
                                      dim=-1).mean().item()
            for i in range(len(hs) - 1)]


def bi_order(mid):
    m = load_fresh(mid)
    bis = block_influence(m, mid, train_text)
    n = len(get_layers(m))
    free(m)
    return sorted(range(n), key=lambda i: bis[i]), bis


def hard_remove(mid, idx):
    m = load_fresh(mid)
    kept = [b for i, b in enumerate(get_layers(m)) if i not in set(idx)]
    set_layers(m, kept)
    reindex(m)
    return m


# ----------------------------------------------------------------------------
# A. KD repair of the pruned model   (the notebook that was never run)
# ----------------------------------------------------------------------------
def kd_repair(student, teacher, mid, steps, lr, gen, T=2.0, alpha=0.5):
    """CE against the data plus T^2-scaled KL against the frozen teacher.

    The T^2 factor is not cosmetic: without it the gradient magnitude of the KL
    term scales as 1/T^2, so raising T to soften the distribution silently
    downweights the distillation signal. This is the standard correction.
    """
    ids = tok[mid](train_text[:40000], return_tensors="pt",
                   add_special_tokens=False)["input_ids"][0]
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    student.train()
    teacher.eval()
    hist = []
    for i in range(steps):
        start = int(torch.randint(0, max(1, len(ids) - SEQ_LEN), (1,),
                                  generator=gen).item())
        x = ids[start:start + SEQ_LEN].unsqueeze(0).to(device)
        s = student(input_ids=x, labels=x, use_cache=False)
        with torch.no_grad():
            t = teacher(input_ids=x, use_cache=False)
        s_log = F.log_softmax(s.logits / T, dim=-1)
        t_p = F.softmax(t.logits / T, dim=-1)
        kl = F.kl_div(s_log, t_p, log_target=False,
                      reduction="batchmean") * (T * T)
        loss = alpha * s.loss + (1 - alpha) * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if i % 25 == 0:
            hist.append((i, round(loss.item(), 4), round(s.loss.item(), 4),
                         round(kl.item(), 5)))
            print(f"      step {i:4d} loss={loss.item():.4f} "
                  f"ce={s.loss.item():.4f} kl={kl.item():.5f}", flush=True)
    student.eval()
    return hist


# ----------------------------------------------------------------------------
# B. LayerDrop training   (approach 11)
# ----------------------------------------------------------------------------
class Switchable(nn.Module):
    """Wraps a block so it can be bypassed without surgery on the ModuleList.

    The original notebook swapped get_layers() contents mid-training and never
    renumbered layer_idx, so every kept block read the wrong position embedding
    on GPT-2. Reordering by hand every step is also slow and error prone. A
    wrapper with a shared mutable mask is neither.
    """

    def __init__(self, block, mask, idx):
        super().__init__()
        self.block = block
        self.mask = mask
        self.idx = idx

    def forward(self, hidden_states, *a, **kw):
        if self.mask[self.idx]:
            return self.block(hidden_states, *a, **kw)
        return hidden_states


def install_switches(m, mask):
    blocks = list(get_layers(m))
    set_layers(m, [Switchable(b, mask, i) for i, b in enumerate(blocks)])
    return blocks


def remove_switches(m, original):
    set_layers(m, original)
    reindex(m)


def layerdrop_train(mid, steps, lr, p, gen):
    m = load_fresh(mid)
    ids = tok[mid](train_text[:40000], return_tensors="pt",
                   add_special_tokens=False)["input_ids"][0]
    mask = [True] * len(get_layers(m))
    original = install_switches(m, mask)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    for i in range(steps):
        # per-step mask from an explicit generator, so the run is reproducible
        r = torch.rand(len(mask), generator=gen).tolist()
        for j in range(len(mask)):
            mask[j] = r[j] > p
        start = int(torch.randint(0, max(1, len(ids) - SEQ_LEN), (1,),
                                  generator=gen).item())
        x = ids[start:start + SEQ_LEN].unsqueeze(0).to(device)
        out = m(input_ids=x, labels=x, use_cache=False)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if i % 20 == 0:
            print(f"      step {i:4d} loss={out.loss.item():.4f} "
                  f"kept={sum(mask)}/{len(mask)}", flush=True)
    remove_switches(m, original)
    m.eval()
    return m


# ----------------------------------------------------------------------------
# C. Learned soft gates   (approach 10)
# ----------------------------------------------------------------------------
class Gated(nn.Module):
    """out = g * block(x) + (1-g) * x, with g = sigmoid(logit).

    This is a SOFT gate trained with a linear pull toward zero, not full L0
    regularisation with a hard-concrete sampler. Stating that plainly matters:
    the gate is a cheap learned ranking signal, and the claim it supports is
    only that learned selection is compared against BI on equal footing.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.logit = nn.Parameter(torch.tensor(3.0))

    def forward(self, hidden_states, *a, **kw):
        g = torch.sigmoid(self.logit)
        out = self.block(hidden_states, *a, **kw)
        if isinstance(out, tuple):
            h = out[0]
            rest = out[1:]
            return (h * g + hidden_states * (1 - g),) + rest
        return out * g + hidden_states * (1 - g)


def gate_train(mid, steps, lr, lam, gen):
    m = load_fresh(mid)
    blocks = list(get_layers(m))
    set_layers(m, [Gated(b) for b in blocks])
    reindex(m)
    ids = tok[mid](train_text[:40000], return_tensors="pt",
                   add_special_tokens=False)["input_ids"][0]
    gates = [b.logit for b in get_layers(m)]
    opt = torch.optim.AdamW(gates, lr=lr)
    m.train()
    for i in range(steps):
        start = int(torch.randint(0, max(1, len(ids) - SEQ_LEN), (1,),
                                  generator=gen).item())
        x = ids[start:start + SEQ_LEN].unsqueeze(0).to(device)
        out = m(input_ids=x, labels=x, use_cache=False)
        penalty = torch.sigmoid(torch.stack(gates)).sum()
        loss = out.loss + lam * penalty
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 20 == 0:
            print(f"      step {i:4d} ce={out.loss.item():.4f} "
                  f"mean_gate={torch.sigmoid(torch.stack(gates)).mean().item():.4f}",
                  flush=True)
    scores = [float(torch.sigmoid(g).item()) for g in gates]
    m.eval()
    return scores


# ----------------------------------------------------------------------------
# D. Attention-head pruning   (approach 12, honestly implemented)
# ----------------------------------------------------------------------------
def out_proj(blk):
    """The attention output projection, across both naming conventions.

    GPT-2 uses attn.c_proj, which is a Conv1D whose weight is [in, out].
    Llama/SmolLM use self_attn.o_proj, an nn.Linear whose weight is [out, in].
    The head axis is therefore on a different dimension in each case, and
    getting that wrong silently prunes nothing.
    """
    sa = getattr(blk, "attn", None) or getattr(blk, "self_attn", None)
    if sa is None:
        return None, None
    for nm in ("o_proj", "c_proj"):
        if hasattr(sa, nm):
            return sa, getattr(sa, nm)
    return None, None


def head_geometry(m):
    cfg = m.config
    nh = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    hs = getattr(cfg, "n_embd", None) or cfg.hidden_size
    return nh, hs // nh


@torch.no_grad()
def head_importance(m, mid, text, n_chars=CALIB_CHARS):
    """Mean |activation| per attention head over calibration text.

    Measured on the INPUT of each block's output projection, because that
    tensor is the per-head concatenation we actually want. Hooking the
    attention module's output instead gives the merged residual stream, which
    is not per-head and made an earlier version of this prune nothing at all.
    """
    nh, hd = head_geometry(m)
    acc, count, handles = {}, 0, []

    def mk(i):
        def pre_hook(mod, inp):
            nonlocal count
            v = inp[0].detach().float()                 # [B, T, nh*hd]
            v = v.reshape(-1, nh, hd).abs().mean(dim=(0, 2))   # [nh]
            if i in acc:
                acc[i] += v
            else:
                acc[i] = v
        return pre_hook

    for i, blk in enumerate(get_layers(m)):
        _, proj = out_proj(blk)
        if proj is not None:
            handles.append(proj.register_forward_pre_hook(mk(i)))

    ids = tok[mid](text[:n_chars], return_tensors="pt",
                   add_special_tokens=False)["input_ids"]
    seq = ids[0]
    pos, steps = 0, 0
    while pos + EVAL_BLOCK <= len(seq):
        m(input_ids=ids[:, pos:pos + EVAL_BLOCK].to(device), use_cache=False)
        pos += EVAL_BLOCK
        steps += 1
    for h in handles:
        h.remove()
    if not acc:
        return {}, nh
    return {i: (v / max(1, steps)).tolist() for i, v in acc.items()}, nh


def prune_heads(mid, frac):
    m = load_fresh(mid)
    imp, nh = head_importance(m, mid, train_text)
    _, hd = head_geometry(m)
    kept_counts = []
    for i, blk in enumerate(get_layers(m)):
        _, proj = out_proj(blk)
        if proj is None or i not in imp:
            continue
        order = np.argsort(imp[i])                     # least important first
        drop = order[:int(len(order) * frac)]
        w = proj.weight
        with torch.no_grad():
            for h in drop:
                if w.dim() == 2 and w.shape[0] == nh * hd:
                    w[h * hd:(h + 1) * hd, :] = 0.0     # Conv1D, [in, out]
                elif w.dim() == 2:
                    w[:, h * hd:(h + 1) * hd] = 0.0     # Linear, [out, in]
                elif w.dim() == 3:
                    w[h, :, :] = 0.0                    # fused qkv layout
        kept_counts.append(len(order) - len(drop))
    m.eval()
    return m, (min(kept_counts) if kept_counts else 0), nh


# ----------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------
def exp_kd(steps, lr, log):
    rows = []
    for mid in MODEL_IDS:
        short = mid.split("/")[-1]
        order, bis = bi_order(mid)
        teacher = load_fresh(mid)
        base = ppl(teacher, mid, val_text)
        for k in KS:
            for seed in SEEDS:
                gen = torch.Generator().manual_seed(1000 + seed)
                torch.manual_seed(1000 + seed)
                m = hard_remove(mid, sorted(order[:k]))
                pre = ppl(m, mid, val_text)
                t0 = time.perf_counter()
                kd_repair(m, teacher, mid, steps, lr, gen)
                dt = time.perf_counter() - t0
                post = ppl(m, mid, val_text)
                rows.append({"approach": "A_kd_repair", "model": short, "k": k,
                             "seed": seed, "target": sorted(order[:k]),
                             "ppl_baseline": base, "ppl_before": pre,
                             "ppl_after": post,
                             "recovered_pct": 100 * (pre - post) / max(1e-9, pre - base),
                             "params": n_params(m), "kd_seconds": round(dt, 1)})
                print(f"  A {short:14s} k={k} s={seed}  {base:.2f} -> "
                      f"{pre:.2f} -> {post:.2f}   recovered "
                      f"{rows[-1]['recovered_pct']:5.1f}%  [{dt:.0f}s]", flush=True)
                free(m)
        free(teacher)
    return pd.DataFrame(rows)


def exp_layerdrop(steps, lr, p, log):
    rows = []
    for mid in MODEL_IDS:
        short = mid.split("/")[-1]
        m0 = load_fresh(mid)
        base = ppl(m0, mid, val_text)
        n = len(get_layers(m0))
        free(m0)
        order, _ = bi_order(mid)
        for seed in SEEDS:
            gen = torch.Generator().manual_seed(2000 + seed)
            t0 = time.perf_counter()
            m = layerdrop_train(mid, steps, lr, p, gen)
            dt = time.perf_counter() - t0
            for k in KS:
                # Remove the BI-least-important blocks, the same targets every
                # other approach uses. Removing blocks 0..k-1 instead would
                # delete GPT-2's first layers, which are among its most
                # important, and that measures the wrong thing entirely.
                mm = hard_remove_from(m, mid, sorted(order[:k]))
                rows.append({"approach": "B_layerdrop", "model": short,
                             "k": k, "seed": seed,
                             "target": sorted(order[:k]),
                             "ppl_baseline": base, "ppl": ppl(mm, mid, val_text),
                             "params": n_params(mm),
                             "train_seconds": round(dt, 1)})
                print(f"  B {short:14s} k={k} s={seed}  base={base:.2f} "
                      f"ppl={rows[-1]['ppl']:.2f}  "
                      f"target={sorted(order[:k])}  [{dt:.0f}s]", flush=True)
                free(mm)
            free(m)
    return pd.DataFrame(rows)


def hard_remove_from(m, mid, idx):
    """Remove blocks from an already-fine-tuned in-memory model."""
    import copy
    mm = copy.deepcopy(m)
    kept = [b for i, b in enumerate(get_layers(mm)) if i not in set(idx)]
    set_layers(mm, kept)
    reindex(mm)
    mm.eval()
    return mm


def exp_gates(steps, lr, lam, log):
    rows = []
    for mid in MODEL_IDS:
        short = mid.split("/")[-1]
        m0 = load_fresh(mid)
        base = ppl(m0, mid, val_text)
        order, bis = bi_order(mid)
        free(m0)
        for seed in SEEDS:
            gen = torch.Generator().manual_seed(3000 + seed)
            scores = gate_train(mid, steps, lr, lam, gen)
            g_order = sorted(range(len(scores)), key=lambda i: scores[i])
            for k in KS:
                m = hard_remove(mid, sorted(g_order[:k]))
                rows.append({"approach": "C_gates", "model": short, "k": k,
                             "seed": seed, "target": sorted(g_order[:k]),
                             "gate_order": g_order,
                             "ppl_baseline": base, "ppl": ppl(m, mid, val_text),
                             "params": n_params(m),
                             "overlap_with_bi": len(set(g_order[:k]) & set(order[:k]))})
                print(f"  C {short:14s} k={k} s={seed}  base={base:.2f} "
                      f"ppl={rows[-1]['ppl']:.2f}  target={sorted(g_order[:k])} "
                      f"BI={sorted(order[:k])}", flush=True)
                free(m)
    return pd.DataFrame(rows)


def exp_heads(frac, log):
    rows = []
    for mid in MODEL_IDS:
        short = mid.split("/")[-1]
        m0 = load_fresh(mid)
        base = ppl(m0, mid, val_text)
        pb = n_params(m0)
        free(m0)
        for f in frac:
            m, nkeep, nh = prune_heads(mid, f)
            rows.append({"approach": "D_head_prune", "model": short,
                         "frac_pruned": f, "heads_total": nh,
                         "heads_kept_worst_block": nkeep,
                         "ppl_baseline": base, "ppl": ppl(m, mid, val_text),
                         "params": pb})  # masking does not change the count
            print(f"  D {short:14s} prune={f:.0%}  base={base:.2f} "
                  f"ppl={rows[-1]['ppl']:.2f}", flush=True)
            free(m)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="ABCD")
    ap.add_argument("--kd-steps", type=int, default=200)
    ap.add_argument("--kd-lr", type=float, default=2e-5)
    ap.add_argument("--drop-steps", type=int, default=60)
    ap.add_argument("--drop-lr", type=float, default=2e-5)
    ap.add_argument("--drop-p", type=float, default=0.2)
    ap.add_argument("--gate-steps", type=int, default=60)
    ap.add_argument("--gate-lr", type=float, default=1e-2)
    ap.add_argument("--gate-lam", type=float, default=0.01)
    ap.add_argument("--head-fracs", default="0.0,0.25,0.5")
    ap.add_argument("--out", default="results/unrun_approaches.csv")
    a = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    log = open("results/unrun_log.txt", "a")
    log.write(f"\n===== run {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"which={a.which} kd_steps={a.kd_steps} =====\n")
    log.flush()
    frames = []

    if "A" in a.which:
        print("\n=== A: KD repair of the BI-pruned model ===", flush=True)
        frames.append(exp_kd(a.kd_steps, a.kd_lr, log))
    if "B" in a.which:
        print("\n=== B: LayerDrop training ===", flush=True)
        frames.append(exp_layerdrop(a.drop_steps, a.drop_lr, a.drop_p, log))
    if "C" in a.which:
        print("\n=== C: learned soft gates ===", flush=True)
        frames.append(exp_gates(a.gate_steps, a.gate_lr, a.gate_lam, log))
    if "D" in a.which:
        print("\n=== D: attention-head pruning ===", flush=True)
        frames.append(exp_heads([float(x) for x in a.head_fracs.split(",")], log))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(a.out, index=False)
    log.write(df.to_string() + "\n")
    log.close()
    print(f"\nwrote {a.out}  ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
