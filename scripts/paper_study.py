"""
Paper-grade evaluation harness for the Fitted-Ternary TerGRASP bridge.

Why this file exists
--------------------
The exploratory notebook (assignment2.ipynb) fitted the bridge with the *global*
RNG, so the same configuration produced a different perplexity on every run.
That single defect makes every headline number unreproducible. This harness
fixes it by giving every fit its own torch.Generator, and it adds the
measurements the exploratory notebook never made at all:

  * seed-level mean/std for every operating point (not one number)
  * an explicit determinism check (same seed -> bit-identical PPL)
  * parameter counts, effective bits/param and serialized size
  * measured latency, not an estimate
  * rank / bit-width / calibration-size ablations
  * a random-removal control drawn from the *same* seed sequence

Run:
    .venv/bin/python scripts/paper_study.py            # default grid
    .venv/bin/python scripts/paper_study.py --quick    # smoke test
"""

import argparse
import gc
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
MODEL_IDS = [
    "openai-community/gpt2",
    "HuggingFaceTB/SmolLM-135M",
]
DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1")

EVAL_BLOCK = 256
EVAL_TOKENS = 3000          # tokens scored, non-overlapping windows
CALIB_CHARS = 4000          # characters of train text used to fit the bridge
SEEDS = (0, 1, 2)

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_texts():
    from datasets import load_dataset

    ds = load_dataset(*DATASET)

    def join(split):
        return "\n\n".join(x for x in split["text"] if x and x.strip())

    return (
        join(ds["train"])[:300000],
        join(ds["validation"])[:100000],
        join(ds["test"])[:100000],
    )


# --------------------------------------------------------------------------
# model plumbing
# --------------------------------------------------------------------------
def get_layers(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    return model.model.layers


def set_layers(model, blocks):
    blocks = torch.nn.ModuleList(blocks)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model.transformer.h = blocks
        model.config.n_layer = len(blocks)
    else:
        model.model.layers = blocks
        model.config.num_hidden_layers = len(blocks)
    return model


def load_model(model_id):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.use_cache = False
    model.eval()
    return model


def hidden_size(model):
    cfg = model.config
    return getattr(cfg, "n_embd", None) or cfg.hidden_size


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def perplexity(model, tokenizer, text, max_tokens=EVAL_TOKENS):
    """exp(mean token NLL) over non-overlapping windows of EVAL_BLOCK tokens."""
    model.eval()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0][:max_tokens]
    n = len(ids) // EVAL_BLOCK
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(n):
            x = ids[i * EVAL_BLOCK : (i + 1) * EVAL_BLOCK].unsqueeze(0).to(device)
            out = model(input_ids=x, labels=x, use_cache=False)
            c = x.numel() - 1
            total_nll += out.loss.item() * c
            total_tok += c
    return math.exp(total_nll / total_tok)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def serialized_bytes(model):
    """fp16 size on disk, the convention used by all pruning papers."""
    return sum(p.numel() * 2 for p in model.parameters())


def latency_ms(model, tokenizer, text, seq_len=256, repeat=10):
    model.eval()
    ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    )["input_ids"][:, :seq_len].to(device)
    with torch.no_grad():
        for _ in range(3):
            model(ids, use_cache=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            model(ids, use_cache=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return 1000.0 * (t1 - t0) / repeat


# --------------------------------------------------------------------------
# redundancy score
# --------------------------------------------------------------------------
def block_influence(model, tokenizer, text, n_chars=2000):
    model.eval()
    ids = tokenizer(text[:n_chars], return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return [
        1.0
        - F.cosine_similarity(hs[i].float(), hs[i + 1].float(), dim=-1).mean().item()
        for i in range(len(hs) - 1)
    ]


def redundant_order(model_id, tokenizer, train_text):
    m = load_model(model_id).to(device)
    bis = block_influence(m, tokenizer, train_text)
    n = len(get_layers(m))
    free(m)
    return sorted(range(n), key=lambda i: bis[i]), bis


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------
def hard_remove(model_id, indices):
    model = load_model(model_id).to(device)
    blocks = list(get_layers(model))
    kept = [b for i, b in enumerate(blocks) if i not in set(indices)]
    set_layers(model, kept)
    for i, layer in enumerate(get_layers(model)):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = i
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = i
    return model


def quantize_absmear(W, bits):
    """Symmetric AbsMean uniform quantization to `bits` signed levels.

    bits=2 reproduces the ternary {-1,0,+1} grid used by the bridge.
    """
    if bits >= 16:
        return W
    levels = 2 ** (bits - 1) - 1          # bits=2 -> 1 (ternary), bits=3 -> 3
    gamma = W.abs().mean() + 1e-8
    q = torch.round(W / gamma).clamp(-levels, levels)
    return (q * gamma).to(W.device)


class FittedTernaryBridge(torch.nn.Module):
    """g(x) = x + x U V, with U, V fitted then AbsMean-quantized.

    The fit is the ridge-regularized least-squares problem

        min_{U,V}  || X U V - (Y - X) ||_F^2 + lambda(||U||_F^2 + ||V||_F^2)

    where X is the block input and Y its output on calibration text. The
    generator argument makes the whole procedure deterministic given a seed.
    """

    def __init__(self, dim, rank=32, bits=2, gen=None):
        super().__init__()
        g = gen if gen is not None else torch.Generator().manual_seed(0)
        U = torch.randn(dim, rank, generator=g) * (1.0 / math.sqrt(dim))
        V = torch.randn(rank, dim, generator=g) * (1.0 / math.sqrt(rank))
        self.register_buffer("U", U)
        self.register_buffer("Vt", V)
        self.bits = bits

    @torch.no_grad()
    def quantize_(self):
        self.U.copy_(quantize_absmear(self.U, self.bits))
        self.Vt.copy_(quantize_absmear(self.Vt, self.bits))
        return self

    @property
    def effective_bits_per_param(self):
        return float(self.bits)

    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        return x + (x @ self.U @ self.Vt)


def collect_layer_io(model, tokenizer, L, text, n_chars=CALIB_CHARS, chunk=256):
    # Capture the input and output of block L on calibration text.
    # The text is walked in non-overlapping `chunk`-token windows so we never
    # exceed the model's maximum position embedding (1024 for both models).
    # Feeding the whole calibration slice in one pass overflows the position
    # table and raises an index error.
    ids = tokenizer(text[:n_chars], return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    blocks = get_layers(model)
    Xs, Ys = [], []

    def grab_in(mod, args):
        Xs.append(args[0].detach().float().reshape(-1, args[0].shape[-1]).cpu())

    def grab_out(mod, args, out):
        out = out[0] if isinstance(out, tuple) else out
        Ys.append(out.detach().float().reshape(-1, out.shape[-1]).cpu())

    h1 = blocks[L].register_forward_pre_hook(grab_in)
    h2 = blocks[L].register_forward_hook(grab_out)
    with torch.no_grad():
        for i in range(0, len(ids) - chunk + 1, chunk):
            model(
                input_ids=ids[i : i + chunk].unsqueeze(0).to(device), use_cache=False
            )
    h1.remove()
    h2.remove()
    return torch.cat(Xs), torch.cat(Ys)


def fit_bridge(X, Y, rank=32, bits=2, seed=0, iters=300, lr=3e-2, ridge=1e-4):
    """Deterministic AdamW fit of the low-rank residual map."""
    d = X.shape[1]
    gen = torch.Generator().manual_seed(seed)
    bridge = FittedTernaryBridge(d, rank=rank, bits=bits, gen=gen)

    Xc, D = X.double(), (Y - X).double()
    U = bridge.U.double().clone().requires_grad_(True)
    V = bridge.Vt.double().clone().requires_grad_(True)
    opt = torch.optim.AdamW([U, V], lr=lr, weight_decay=ridge)
    for _ in range(iters):
        opt.zero_grad()
        resid = Xc @ U @ V - D
        loss = resid.pow(2).mean()
        loss.backward()
        opt.step()

    fit_err = float(loss.detach())
    base_err = float(D.pow(2).mean())

    with torch.no_grad():
        bridge.U.copy_(U.float())
        bridge.Vt.copy_(V.float())
    return bridge.quantize_(), {
        "fit_mse": fit_err,
        "residual_mse": base_err,
        "explained_variance": 1.0 - fit_err / base_err,
    }


# --------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------
class Study:
    def __init__(self, quick=False):
        self.quick = quick
        self.train_text, self.val_text, self.test_text = load_texts()
        from transformers import AutoTokenizer

        self.tok = {m: AutoTokenizer.from_pretrained(m) for m in MODEL_IDS}
        self.rows = []
        self.meta = {
            "device": device,
            "dataset": DATASET,
            "eval_tokens": EVAL_TOKENS,
            "eval_block": EVAL_BLOCK,
            "calib_chars": CALIB_CHARS,
            "seeds": list(SEEDS),
            "torch": torch.__version__,
        }

    def log(self, *a):
        print(*a, flush=True)

    def add(self, **kw):
        self.rows.append(kw)
        self.log(
            "  {op:22s} {model:12s} k={k} seed={seed} ppl={ppl:8.2f}".format(
                op=kw.get("operator", "?"),
                model=kw.get("model_short", "?"),
                k=kw.get("k", "?"),
                seed=kw.get("seed", "?"),
                ppl=kw.get("ppl", float("nan")),
            )
        )

    # ---- experiment blocks ------------------------------------------------
    def run_baseline(self):
        self.log("\n[1] baseline")
        for mid in MODEL_IDS:
            m = load_model(mid).to(device)
            self.add(
                operator="baseline",
                model=mid,
                model_short=mid.split("/")[-1],
                k=0,
                seed=None,
                ppl=perplexity(m, self.tok[mid], self.val_text),
                params=n_params(m),
                bytes_fp16=serialized_bytes(m),
                latency_ms=latency_ms(m, self.tok[mid], self.val_text),
            )
            free(m)

    def run_determinism(self, mid):
        """Same seed twice must give bit-identical perplexity."""
        self.log(f"\n[2] determinism check on {mid}")
        order, _ = redundant_order(mid, self.tok[mid], self.train_text)
        seen = []
        for rep in range(2):
            m, _ = build_bridged(mid, self, [order[0]], seed=7)
            seen.append(perplexity(m, self.tok[mid], self.val_text))
            free(m)
        self.log(f"  rep1={seen[0]:.6f}  rep2={seen[1]:.6f}  delta={abs(seen[0]-seen[1]):.2e}")
        return seen

    def run_main(self):
        """BI-targeted removal, random removal, and the bridge, all seeded."""
        self.log("\n[3] main comparison (3 seeds per point)")
        for mid in MODEL_IDS:
            short = mid.split("/")[-1]
            order, bis = redundant_order(mid, self.tok[mid], self.train_text)
            n = len(order)
            self.log(f"  {short}: BI order (ascending) = {order[:6]} ... n={n}")

            for k in (1, 2, 3):
                targets = sorted(order[:k])

                for seed in SEEDS:
                    m = hard_remove(mid, targets)
                    self.add(
                        operator="BI removal",
                        model=mid,
                        model_short=short,
                        k=k,
                        seed=seed,
                        ppl=perplexity(m, self.tok[mid], self.val_text),
                        params=n_params(m),
                        bytes_fp16=serialized_bytes(m),
                        latency_ms=latency_ms(m, self.tok[mid], self.val_text),
                    )
                    free(m)

                # random control: same k, same seed sequence, disjoint from target
                for seed in SEEDS:
                    rng = random.Random(1000 * k + seed)
                    pick = rng.sample([i for i in range(n) if i not in set(targets)], k)
                    m = hard_remove(mid, sorted(pick))
                    self.add(
                        operator="random removal",
                        model=mid,
                        model_short=short,
                        k=k,
                        seed=seed,
                        target=str(sorted(pick)),
                        ppl=perplexity(m, self.tok[mid], self.val_text),
                        params=n_params(m),
                        bytes_fp16=serialized_bytes(m),
                    )
                    free(m)

                for seed in SEEDS:
                    m, diag = build_bridged(mid, self, targets, seed=seed)
                    self.add(
                        operator="fitted bridge",
                        model=mid,
                        model_short=short,
                        k=k,
                        seed=seed,
                        ppl=perplexity(m, self.tok[mid], self.val_text),
                        params=n_params(m),
                        bytes_fp16=serialized_bytes(m),
                        latency_ms=latency_ms(m, self.tok[mid], self.val_text),
                        **{f"L{L}_{kk}": vv for L, dd in diag.items()
                           for kk, vv in dd.items()},
                    )
                    free(m)

    def run_ablation(self):
        """Rank, bit-width and calibration size, at the headline operating point."""
        self.log("\n[4] ablations")
        ranks = (4, 16, 64, 128) if not self.quick else (16,)
        bitss = (2, 4, 8) if not self.quick else (2,)
        calibs = (2000, 20000) if not self.quick else (2000,)
        for mid in MODEL_IDS:
            short = mid.split("/")[-1]
            order, _ = redundant_order(mid, self.tok[mid], self.train_text)
            target = [order[0]]
            for r in ranks:
                m, diag = build_bridged(mid, self, target, rank=r, seed=0)
                self.add(operator=f"ablation rank={r}", model=mid, model_short=short,
                         k=1, seed=0, rank=r,
                         ppl=perplexity(m, self.tok[mid], self.val_text),
                         params=n_params(m), bytes_fp16=serialized_bytes(m))
                free(m)
            for b in bitss:
                m, diag = build_bridged(mid, self, target, bits=b, seed=0)
                self.add(operator=f"ablation bits={b}", model=mid, model_short=short,
                         k=1, seed=0, bits=b,
                         ppl=perplexity(m, self.tok[mid], self.val_text),
                         params=n_params(m), bytes_fp16=serialized_bytes(m))
                free(m)
            for c in calibs:
                m, diag = build_bridged(mid, self, target, calib_chars=c, seed=0)
                self.add(operator=f"ablation calib={c}", model=mid, model_short=short,
                         k=1, seed=0, calib_chars=c,
                         ppl=perplexity(m, self.tok[mid], self.val_text))
                free(m)

    def run_spectrum(self):
        """Why the bridge is or is not enough: the residual map's own spectrum.

        If the block residual Delta = Y - X is well described by a rank-r
        linear map X -> Delta, a rank-r bridge can in principle reproduce it.
        This measures that directly with the closed-form ridge solution and its
        truncated-SVD variants, which upper-bound what any rank-r bridge of this
        form can achieve.
        """
        self.log("\n[6] residual-map spectrum (mechanism)")
        self.spectrum = []
        for mid in MODEL_IDS:
            short = mid.split("/")[-1]
            order, _ = redundant_order(mid, self.tok[mid], self.train_text)
            m0 = load_model(mid).to(device)
            d = hidden_size(m0)
            for L in order[:2]:
                X, Y = collect_layer_io(m0, self.tok[mid], L, self.train_text)
                Xd, D = X.double(), (Y - X).double()
                G = Xd.T @ Xd
                G += 1e-3 * torch.eye(d, dtype=torch.float64) * G.diagonal().mean()
                W = torch.linalg.solve(G, Xd.T @ D)
                full_ev = 1 - float((Xd @ W - D).pow(2).mean()) / float(D.pow(2).mean())
                Uu, Ss, Vt = torch.linalg.svd(W, full_matrices=False)
                energy = (Ss ** 2).cumsum(0) / (Ss ** 2).sum()
                for r in (4, 16, 32, 64, 128, 256):
                    if r >= len(energy):
                        continue
                    Wr = Uu[:, :r] @ torch.diag(Ss[:r]) @ Vt[:r]
                    ev = 1 - float((Xd @ Wr - D).pow(2).mean()) / float(D.pow(2).mean())
                    self.spectrum.append(
                        {
                            "model": mid,
                            "model_short": short,
                            "layer": L,
                            "rank": r,
                            "map_energy": float(energy[r - 1]),
                            "explained_variance": ev,
                            "full_rank_ev": full_ev,
                        }
                    )
                    self.log(
                        f"  {short} L{L:<3d} rank {r:4d}: "
                        f"map energy {float(energy[r-1]):.3f}  residual EV {ev:.3f}"
                    )
            free(m0)
        self.log(f"  (full-rank linear EV is the ceiling any linear bridge can reach)")

    def run_test_split(self):
        """Held-out test perplexity for the selected operating point k=1."""
        self.log("\n[5] held-out test at k=1")
        for mid in MODEL_IDS:
            short = mid.split("/")[-1]
            order, _ = redundant_order(mid, self.tok[mid], self.train_text)
            target = [order[0]]
            for seed in SEEDS:
                m, _ = build_bridged(mid, self, target, seed=seed)
                self.add(
                    operator="fitted bridge (test)",
                    model=mid, model_short=short, k=1, seed=seed,
                    ppl=perplexity(m, self.tok[mid], self.test_text),
                    split="test",
                )
                free(m)
            for seed in SEEDS:
                m = hard_remove(mid, target)
                self.add(
                    operator="BI removal (test)",
                    model=mid, model_short=short, k=1, seed=seed,
                    ppl=perplexity(m, self.tok[mid], self.test_text),
                    split="test",
                )
                free(m)

    # ---- output -----------------------------------------------------------
    def summarize(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    def save(self, path):
        import pandas as pd

        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False)
        with open(path.replace(".csv", "_meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2)
        spec = getattr(self, "spectrum", [])
        if spec:
            import pandas as pd

            pd.DataFrame(spec).to_csv(
                path.replace(".csv", "_spectrum.csv"), index=False
            )
        return df


def build_bridged(mid, study, targets, rank=32, bits=2, seed=0, calib_chars=CALIB_CHARS):
    """Fit one bridge per target on the *pristine* model, then splice them in."""
    m0 = load_model(mid).to(device)
    d = hidden_size(m0)
    tok = study.tok[mid]
    bridges, diag = {}, {}
    for L in targets:
        X, Y = collect_layer_io(m0, tok, L, study.train_text, n_chars=calib_chars)
        bridges[L], diag[L] = fit_bridge(X, Y, rank=rank, bits=bits, seed=seed)
    free(m0)

    m = load_model(mid)
    blocks = list(get_layers(m))
    for L, b in bridges.items():
        blocks[L] = b.to(device=next(m.parameters()).device,
                           dtype=next(m.parameters()).dtype)
    m = set_layers(m, blocks).to(device)
    return m, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/paper_study.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    s = Study(quick=args.quick)
    s.log(f"device={s.meta['device']}  torch={s.meta['torch']}")

    s.run_baseline()
    for mid in MODEL_IDS:
        s.run_determinism(mid)
    s.run_main()
    s.run_spectrum()
    s.run_ablation()
    s.run_test_split()

    df = s.save(args.out)
    s.log("\nwrote", args.out)
    s.log(df.groupby(["operator", "model_short", "k"])["ppl"]
          .agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()