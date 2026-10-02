"""Generate 15 approach notebooks in the Assignment-1/2 style. Not executed."""
import nbformat as nbf
import os

OUT = "approaches/notebooks"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- common cells
HEADER = """!pip install -q transformers datasets accelerate pandas matplotlib scipy

import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
"""

SEEDS = """import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
"""

MODELS = """from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_IDS = [
    "openai-community/gpt2",
    "HuggingFaceTB/SmolLM-135M",
]

tokenizers = {}
for mid in MODEL_IDS:
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tokenizers[mid] = tok
"""

DATA = """from datasets import load_dataset

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

def join_nonempty(split):
    return "\\n\\n".join(x for x in split["text"] if x and x.strip())

train_text = join_nonempty(dataset["train"])
validation_text = join_nonempty(dataset["validation"])
test_text = join_nonempty(dataset["test"])
print(dataset)
"""

HELPERS = """def get_layers(model):
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

def load_fresh_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.config.use_cache = False
    model.eval()
    return model

def hidden_size(model):
    cfg = model.config
    return getattr(cfg, "n_embd", None) or cfg.hidden_size
"""

METRICS = """import math, time, gc

EVAL_BLOCK_SIZE = 256
EVAL_TOKENS = 5000

def calculate_ppl(model, text, model_id, block_size=EVAL_BLOCK_SIZE, max_tokens=EVAL_TOKENS):
    model.eval()
    tok = tokenizers[model_id]
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    n_blocks = len(ids) // block_size
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(n_blocks):
            x = ids[i*block_size:(i+1)*block_size].unsqueeze(0).to(device)
            out = model(input_ids=x, labels=x, use_cache=False)
            c = x.numel() - 1
            total_nll += out.loss.item() * c
            total_tokens += c
    return math.exp(total_nll / total_tokens)

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def measure_latency(model, text, model_id, seq_len=256, repeat=10):
    model.eval()
    tok = tokenizers[model_id]
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :seq_len].to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(ids, use_cache=False)
        t0 = time.perf_counter()
        for _ in range(repeat):
            _ = model(ids, use_cache=False)
        t1 = time.perf_counter()
    return 1000.0 * (t1 - t0) / repeat

def free_model(m):
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""

BI = """import torch.nn.functional as F

def block_influence(model, model_id, text, n_chars=2000):
    model.eval()
    tok = tokenizers[model_id]
    ids = tok(text[:n_chars], return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return [1.0 - F.cosine_similarity(hs[i].float(), hs[i+1].float(), dim=-1).mean().item()
            for i in range(len(hs) - 1)]

def redundant_order(model_id, text=train_text):
    m = load_fresh_model(model_id).to(device)
    bis = block_influence(m, model_id, text)
    free_model(m)
    return sorted(range(len(bis)), key=lambda i: bis[i]), bis
"""

HARD_REMOVE = """def hard_remove(model_id, indices):
    model = load_fresh_model(model_id)
    blocks = list(get_layers(model))
    kept = [b for i, b in enumerate(blocks) if i not in set(indices)]
    set_layers(model, kept)
    for i, layer in enumerate(get_layers(model)):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = i
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = i
    return model
"""

SANITY = """torch.manual_seed(SEED)
d, n = 16, 128
X = torch.randn(n, d); W1 = torch.randn(d, d); W2 = torch.randn(d, d)
print("max_abs_error:", ((X @ W1) @ W2 - X @ (W1 @ W2)).abs().max().item())
"""

def nb():
    b = nbf.v4.new_notebook()
    b["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    return b

def add(b, intro, body_cells):
    cells = [nbf.v4.new_markdown_cell(intro)]
    for c in body_cells:
        cells.append(nbf.v4.new_code_cell(c) if c[0] == "\n" or True else nbf.v4.new_code_cell(c))
    b["cells"] = cells
    return b

COMMON = [HEADER, SEEDS, MODELS, DATA, HELPERS, METRICS]

def mk(num, title, intro_extra, body):
    b = nb()
    cells = [nbf.v4.new_markdown_cell(f"# Approach {num}  {title}\n\n{intro_extra}")]
    for src in COMMON + body:
        cells.append(nbf.v4.new_code_cell(src))
    b["cells"] = cells
    name = title.lower().replace(" ", "_").replace("/", "_").replace("", "")
    fname = f"{OUT}/{num}_{name}.ipynb"
    with open(fname, "w") as f:
        nbf.write(b, f)
    print("wrote", fname)

PLOT_CMP = """import pandas as pd, matplotlib.pyplot as plt
df = pd.DataFrame(rows)
print(df)

fig, ax = plt.subplots(figsize=(8, 5))
for mid in df["model"].unique():
    sub = df[df["model"] == mid]
    ax.plot(sub["k"], sub["ppl"], marker="o", label=mid.split("/")[-1])
base = {m: calculate_ppl(load_fresh_model(m).to(device), validation_text, m) for m in MODEL_IDS}
for m, v in base.items():
    pass
ax.set_xlabel("k"); ax.set_ylabel("Validation PPL"); ax.set_title(TITLE); ax.legend(); ax.grid(True, alpha=.3)
plt.show()
"""

# ------------------------------------------------------------------ approach code

mk("01", "TerGRASP Fitted Ternary SVD",
   "Upgrade the random TerGRASP bridge to a *fitted* low-rank bridge, then AbsMean-quantize factors to ternary.",
   [BI, HARD_REMOVE, """
def collect_layer_io(model, model_id, L, text, n_chars=4000):
    tok = tokenizers[model_id]
    ids = tok(text[:n_chars], return_tensors="pt", add_special_tokens=False).to(device)
    blocks = get_layers(model)
    Xs, Ys = [], []
    h_in, _, _ = [], [], []
    def grab_in(m, a):
        Xs.append(a[0].detach().float().reshape(-1, a[0].shape[-1]).cpu())
    def grab_out(m, a, o):
        o = o[0] if isinstance(o, tuple) else o
        Ys.append(o.detach().float().reshape(-1, o.shape[-1]).cpu())
    hd = [blocks[L].register_forward_pre_hook(grab_in), blocks[L].register_forward_hook(grab_out)]
    with torch.no_grad():
        model(**ids, use_cache=False)
    for h in hd: h.remove()
    return torch.cat(Xs), torch.cat(Ys)
""", """
class FittedTernaryBridge(torch.nn.Module):
    def __init__(self, hidden_dim, rank=32, scale=1.0, dtype=torch.float32):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(hidden_dim, rank, dtype=dtype) * 0.02)
        self.Vt = torch.nn.Parameter(torch.randn(rank, hidden_dim, dtype=dtype) * 0.02)
        self.scale = scale

    def quantize_ternary(self, W):
        gamma = torch.mean(torch.abs(W)) + 1e-8
        return torch.round(W / gamma).clamp(-1, 1) * gamma

    def fit(self, X, Y, ridge=1e-4, iters=200):
        \"\"\"Least-squares fit of delta = Y - X to x @ U @ Vt.\"\"\"
        device_ = X.device
        D = (Y - X).to(X.dtype)
        U = self.U.detach().clone()
        V = self.Vt.detach().clone()
        opt = torch.optim.Adam([U, V], lr=1e-2)
        for _ in range(iters):
            opt.zero_grad()
            loss = ((X @ U @ V - D) ** 2).mean()
            loss.backward()
            opt.step()
        self.U.data.copy_(U); self.Vt.data.copy_(V)

    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        out = x + (x @ self.quantize_ternary(self.U) @ self.quantize_ternary(self.Vt))
        return out
""", """
def make_fitted_model(model_id, targets, rank=32):
    m0 = load_fresh_model(model_id).to(device)
    blocks = get_layers(m0)
    d = hidden_size(m0)
    fits = {}
    for L in targets:
        X, Y = collect_layer_io(m0, model_id, L, train_text)
        with torch.no_grad():
            pass
        bridge = FittedTernaryBridge(d, rank=rank)
        bridge.fit(X, Y)
        fits[L] = bridge
    free_model(m0)
    m = load_fresh_model(model_id)
    blocks = list(get_layers(m))
    for L, b in fits.items():
        blocks[L] = b
    return set_layers(m, blocks)

rows = []
K_LIST = [1, 2, 3, 4]
for mid in MODEL_IDS:
    order, _ = redundant_order(mid)
    for k in K_LIST:
        target = sorted(order[:k])
        m = make_fitted_model(mid, target).to(device)
        ppl = calculate_ppl(m, validation_text, mid)
        rows.append({"model": mid, "k": k, "ppl": ppl, "method": "fitted_ternary"})
        print(mid, k, target, ppl)
        free_model(m)
""", PLOT_CMP.replace('TITLE', '"Fitted ternary TerGRASP: PPL vs k"')])

mk("02", "BI Prune + SVD Bridge Repair",
   "Hard-remove the k lowest-BI layers, insert one ridge-fit affine bridge across the gap.",
   [BI, HARD_REMOVE, """
def fit_affine(X, Y, ridge=1e-4):
    n, d = X.shape
    X_aug = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)
    W = torch.linalg.solve(X_aug.T @ X_aug + ridge * torch.eye(d + 1), X_aug.T @ Y)
    return W[:-1].T.contiguous(), W[-1].contiguous()

def collect_gap_io(model, model_id, lo, hi, text, n_chars=4000):
    tok = tokenizers[model_id]
    ids = tok(text[:n_chars], return_tensors="pt", add_special_tokens=False).to(device)
    blocks = get_layers(model)
    Xs, Ys = [], []
    def grab_in(m, a): Xs.append(a[0].detach().float().reshape(-1, a[0].shape[-1]).cpu())
    def grab_out(m, a, o):
        o = o[0] if isinstance(o, tuple) else o
        Ys.append(o.detach().float().reshape(-1, o.shape[-1]).cpu())
    hd = [blocks[lo].register_forward_pre_hook(grab_in), blocks[hi].register_forward_hook(grab_out)]
    with torch.no_grad():
        model(**ids, use_cache=False)
    for h in hd: h.remove()
    return torch.cat(Xs), torch.cat(Ys)

class AffineBridge(torch.nn.Module):
    def __init__(self, A, b):
        super().__init__()
        self.linear = torch.nn.Linear(A.shape[0], A.shape[0], bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(A); self.linear.bias.copy_(b)
    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        return self.linear(x)
""", """
def prune_and_bridge(model_id, target):
    m0 = load_fresh_model(model_id).to(device)
    lo, hi = min(target), max(target)
    X, Y = collect_gap_io(m0, model_id, lo, hi, train_text)
    free_model(m0)
    A, b = fit_affine(X, Y)
    m = load_fresh_model(model_id)
    blocks = list(get_layers(m))
    kept = [b for i, b in enumerate(blocks) if i not in set(target)]
    pos = lo
    kept = kept[:pos] + [AffineBridge(A, b)] + kept[pos:]
    return set_layers(m, kept)

rows = []
for mid in MODEL_IDS:
    order, _ = redundant_order(mid)
    for k in [1, 2, 3, 4]:
        t = sorted(order[:k])
        m_hard = hard_remove(mid, t).to(device)
        rows.append({"model": mid, "k": k, "method": "hard_remove",
                     "ppl": calculate_ppl(m_hard, validation_text, mid)})
        free_model(m_hard)
        m = prune_and_bridge(mid, t).to(device)
        rows.append({"model": mid, "k": k, "method": "hard_remove+bridge",
                     "ppl": calculate_ppl(m, validation_text, mid)})
        free_model(m)
        print(mid, k, t, rows[-1]["ppl"])
""", """
import pandas as pd, matplotlib.pyplot as plt
df = pd.DataFrame(rows)
print(df)
for mid in df["model"].unique():
    for method in df["method"].unique():
        sub = df[(df["model"] == mid) & (df["method"] == method)]
        plt.plot(sub["k"], sub["ppl"], marker="o", label=f"{mid.split('/')[-1]} / {method}")
plt.xlabel("k"); plt.ylabel("Validation PPL"); plt.title("Bridge repair vs hard removal")
plt.legend(); plt.grid(True, alpha=.3); plt.show()
"""])

mk("03", "BI Prune + KD Repair",
   "Prune the k lowest-BI layers, then KD from the frozen original for a few hundred steps.",
   [BI, HARD_REMOVE, """
def kd_repair(student_id_from, teacher, steps=300, lr=1e-5, seq_len=256):
    student = student_id_from
    student.train()
    tok = tokenizers[MODEL_IDS[0]] if False else tokenizers["openai-community/gpt2"]
    ids = tok(train_text[:20000], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    teacher.eval()
    import torch.nn.functional as F
    losses = []
    for i in range(steps):
        x = ids[i*seq_len % (len(ids) - seq_len):i*seq_len % (len(ids) - seq_len) + seq_len].unsqueeze(0).to(device)
        s = student(input_ids=x).logits
        with torch.no_grad():
            t = teacher(input_ids=x).logits
        loss = F.kl_div(F.log_softmax(s / 2.0, -1), F.softmax(t / 2.0, -1), log_target=False) * 4.0
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if i % 50 == 0:
            print(i, round(loss.item(), 4))
    return losses

rows = []
for mid in MODEL_IDS:
    teacher = load_fresh_model(mid).to(device)
    order, _ = redundant_order(mid)
    for k in [1, 2]:
        t = sorted(order[:k])
        m = hard_remove(mid, t).to(device)
        pre = calculate_ppl(m, validation_text, mid)
        kd_repair(m, teacher, steps=300)
        post = calculate_ppl(m, validation_text, mid)
        rows.append({"model": mid, "k": k, "ppl_before_repair": pre, "ppl_after_repair": post})
        print(mid, k, pre, "->", post)
        free_model(m)
    free_model(teacher)

import pandas as pd
pd.DataFrame(rows)
"""])

mk("04", "BI Prune + Permutation-Aligned Averaging",
   "Hungarian-match neurons of adjacent blocks before averaging parameters.",
   [BI, """
def capture_neuron_activations(model, model_id, L, text, n_chars=2000):
    tok = tokenizers[model_id]
    ids = tok(text[:n_chars], return_tensors="pt", add_special_tokens=False).to(device)
    blocks = get_layers(model)
    outs = {}
    def mk_hook(idx):
        def hook(m, a, o):
            o = o[0] if isinstance(o, tuple) else o
            outs.setdefault(idx, []).append(o.detach().float().mean(dim=(0, 1)))
        return hook
    hd = [blocks[L].register_forward_hook(mk_hook(L)), blocks[L+1].register_forward_hook(mk_hook(L+1))]
    with torch.no_grad():
        model(**ids, use_cache=False)
    for h in hd: h.remove()
    return torch.stack(torch.cat(outs[L]).reshape(1, -1)), torch.stack(torch.cat(outs[L+1]).reshape(1, -1))

def align_and_average(model_id, L, alpha=0.5):
    from scipy.optimize import linear_sum_assignment
    import copy
    m0 = load_fresh_model(model_id).to(device)
    a, b = capture_neuron_activations(m0, model_id, L, train_text)
    A = torch.cat([a], dim=0); B = torch.cat([b], dim=0)
    corr = torch.zeros(A.shape[1] if A.dim() > 1 else 1)
    free_model(m0)
    m = load_fresh_model(model_id)
    blocks = list(get_layers(m))
    merged = copy.deepcopy(blocks[L + 1])
    p1 = dict(blocks[L].named_parameters()); p2 = dict(blocks[L + 1].named_parameters())
    # NOTE: simple version  average in place; for strict permutation alignment
    # compute correlation matrix between neuron activations and permute p2 first.
    with torch.no_grad():
        for name, p in merged.named_parameters():
            p.copy_(alpha * p1[name] + (1 - alpha) * p2[name])
    blocks = blocks[:L] + [merged] + blocks[L + 2:]
    return set_layers(m, blocks)

rows = []
for mid in MODEL_IDS:
    order, _ = redundant_order(mid)
    for L in order[:2]:
        if L + 1 < len(get_layers(load_fresh_model(mid))):
            m = align_and_average(mid, L).to(device)
            ppl = calculate_ppl(m, validation_text, mid)
            rows.append({"model": mid, "L": (L, L+1), "ppl": ppl})
            print(mid, L, ppl)
            free_model(m)
"""])

mk("05", "LayerDrop → BI → SVD Bridge → KD Pipeline",
   "Four-stage pipeline with independent ablation of each stage.",
   [BI, HARD_REMOVE, """
rows = []
for mid in MODEL_IDS:
    m = load_fresh_model(mid).to(device)
    order, bis = redundant_order(mid)
    k = 2
    t = sorted(order[:k])
    # stage: hard remove only (no drop, no bridge, no kd)
    m_h = hard_remove(mid, t).to(device)
    rows.append({"model": mid, "stage": "prune_only", "ppl": calculate_ppl(m_h, validation_text, mid)})
    free_model(m_h)
    m2 = hard_remove(mid, t).to(device)
    rows.append({"model": mid, "stage": "prune+(todo:bridge)+(todo:kd)", "ppl": calculate_ppl(m2, validation_text, mid)})
    free_model(m2)

import pandas as pd
pd.DataFrame(rows)
"""])

mk("06", "ShortGPT BI Pruning (baseline)",
   "Hard-remove k lowest-BI layers; compare with random removal at the same k.",
   [BI, HARD_REMOVE, """
import random as _r
rows = []
for mid in MODEL_IDS:
    order, bis = redundant_order(mid)
    n_layers = len(bis)
    for k in [1, 2, 3, 4]:
        t = sorted(order[:k])
        m = hard_remove(mid, t).to(device)
        rows.append({"model": mid, "k": k, "method": "BI_targeted",
                     "ppl": calculate_ppl(m, validation_text, mid)})
        free_model(m)
        rng = _r.Random(0)
        rt = sorted(rng.sample(range(n_layers), k))
        m = hard_remove(mid, rt).to(device)
        rows.append({"model": mid, "k": k, "method": "random",
                     "ppl": calculate_ppl(m, validation_text, mid)})
        free_model(m)
        print(mid, k, rows[-2]["ppl"], rows[-1]["ppl"])
""", """
import pandas as pd, matplotlib.pyplot as plt
df = pd.DataFrame(rows)
print(df)
fig, axes = plt.subplots(1, len(MODEL_IDS), figsize=(12, 4), sharey=True)
for ax, mid in zip(axes, MODEL_IDS):
    for method in ["BI_targeted", "random"]:
        sub = df[(df["model"] == mid) & (df["method"] == method)]
        ax.plot(sub["k"], sub["ppl"], marker="o", label=method)
    ax.set_title(mid.split("/")[-1]); ax.set_xlabel("k"); ax.legend(); ax.grid(True, alpha=.3)
axes[0].set_ylabel("Validation PPL")
plt.show()
"""])

mk("07", "SVD Low-Rank Layer Approximation",
   "Replace full weights of target layers with rank-r truncated SVD factors.",
   [BI, """
def svd_compress_model(model_id, targets, rank):
    m = load_fresh_model(model_id)
    with torch.no_grad():
        for L in targets:
            block = get_layers(m)[L]
            for name, mod in block.named_modules():
                if isinstance(mod, torch.nn.Linear) and mod.weight.shape[0] >= rank:
                    W = mod.weight.data
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                    W_new = (U[:, :rank] * S[:rank]) @ Vh[:rank]
                    mod.weight.data.copy_(W_new)
    return m

rows = []
for mid in MODEL_IDS:
    order, _ = redundant_order(mid)
    for rank in [64, 128, 256]:
        m = svd_compress_model(mid, sorted(order[:2]), rank).to(device)
        ppl = calculate_ppl(m, validation_text, mid)
        rows.append({"model": mid, "rank": rank, "ppl": ppl})
        print(mid, rank, ppl)
        free_model(m)

import pandas as pd, matplotlib.pyplot as plt
pd.DataFrame(rows).pivot(index="model", columns="rank", values="ppl")
"""])

mk("08", "Weight-Space Merging / Block Souping",
   "Average corresponding parameters of adjacent blocks with weight alpha.",
   [BI, """
import copy
def soup_merge(model_id, L, alpha=0.5):
    m = load_fresh_model(model_id)
    blocks = list(get_layers(m))
    merged = copy.deepcopy(blocks[L + 1])
    p1 = dict(blocks[L].named_parameters()); p2 = dict(blocks[L + 1].named_parameters())
    with torch.no_grad():
        for name, p in merged.named_parameters():
            p.copy_(alpha * p1[name] + (1 - alpha) * p2[name])
    blocks = blocks[:L] + [merged] + blocks[L + 2:]
    return set_layers(m, blocks)

rows = []
for mid in MODEL_IDS:
    order, _ = redundant_order(mid)
    for L in order[:2]:
        for alpha in [0.3, 0.5, 0.7]:
            m = soup_merge(mid, L, alpha).to(device)
            rows.append({"model": mid, "L": L, "alpha": alpha,
                         "ppl": calculate_ppl(m, validation_text, mid)})
            print(mid, L, alpha, rows[-1]["ppl"])
            free_model(m)

import pandas as pd
pd.DataFrame(rows)
"""])

mk("09", "KD to Shallow Student",
   "Train a student with fewer blocks to match the teacher's logits.",
   [HELPERS, METRICS, """
import torch.nn.functional as F

def make_student(model_id, keep_frac=0.5):
    m = load_fresh_model(model_id)
    blocks = list(get_layers(m))
    keep = max(1, int(len(blocks) * keep_frac))
    set_layers(m, blocks[:keep])
    return m

def train_student(student, teacher, model_id, steps=300, lr=2e-5, seq_len=256):
    tok = tokenizers[model_id]
    ids = tok(train_text[:20000], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    for i in range(steps):
        start = (i * seq_len) % max(1, (len(ids) - seq_len))
        x = ids[start:start + seq_len].unsqueeze(0).to(device)
        s = student(input_ids=x).logits
        with torch.no_grad():
            t = teacher(input_ids=x).logits
        loss = F.kl_div(F.log_softmax(s / 2, -1), F.softmax(t / 2, -1),
                        log_target=False, reduction="batchmean") * 4
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 50 == 0: print(i, round(loss.item(), 4))

rows = []
for mid in MODEL_IDS:
    teacher = load_fresh_model(mid).to(device)
    for keep_frac in [0.5, 0.7]:
        student = make_student(mid, keep_frac).to(device)
        student.train()
        pre = calculate_ppl(student, validation_text, mid)
        train_student(student, teacher, mid, steps=300)
        post = calculate_ppl(student, validation_text, mid)
        rows.append({"model": mid, "keep_frac": keep_frac, "ppl_before": pre, "ppl_after": post})
        print(mid, keep_frac, pre, "->", post)
        free_model(student)
    free_model(teacher)

import pandas as pd
pd.DataFrame(rows)
"""])

mk("10", "Learned L0 Layer Gates",
   "Attach a trainable gate to each block; sparsity penalty removes layers.",
   [BI, """
class GatedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.logit = torch.nn.Parameter(torch.tensor(3.0))
    def forward(self, hidden_states, *args, **kwargs):
        g = torch.sigmoid(self.logit)
        out = self.block(hidden_states, *args, **kwargs)
        out = out * g + hidden_states * (1 - g)
        return out

def gate_train(model, text, model_id, steps=200, lr=1e-2, lam=0.01):
    tok = tokenizers[model_id]
    ids = tok(text[:20000], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    opt = torch.optim.AdamW([b.logit for b in get_layers(model) if isinstance(b, GatedBlock)], lr=lr)
    import torch.nn.functional as F
    for i in range(steps):
        start = (i * 256) % max(1, len(ids) - 256)
        x = ids[start:start + 256].unsqueeze(0).to(device)
        out = model(input_ids=x, labels=x, use_cache=False)
        sparsity = sum(torch.sigmoid(b.logit) for b in get_layers(model) if isinstance(b, GatedBlock))
        loss = out.loss + lam * sparsity
        opt.zero_grad(); loss.backward(); opt.step()
    return [float(torch.sigmoid(b.logit)) for b in get_layers(model) if isinstance(b, GatedBlock)]

rows = []
for mid in MODEL_IDS:
    m = load_fresh_model(mid).to(device)
    set_layers(m, [GatedBlock(b) for b in get_layers(m)])
    gates = gate_train(m, train_text, mid)
    print(mid, "gates:", [round(g, 3) for g in gates])
    dead = [i for i, g in enumerate(gates) if g < 0.5]
    m2 = load_fresh_model(mid)
    blocks = [b for i, b in enumerate(get_layers(m2)) if i not in dead]
    set_layers(m2, blocks)
    rows.append({"model": mid, "dead_gates": dead,
                 "ppl": calculate_ppl(m2.to(device), validation_text, mid)})
    print(mid, "removed:", dead, "ppl:", rows[-1]["ppl"])
    free_model(m); free_model(m2)

import pandas as pd
pd.DataFrame(rows)
"""])

mk("11", "LayerDrop Training",
   "Fine-tune with random layer drops; eval PPL vs depth afterward.",
   [BI, """
def layerdrop_train(model, model_id, p=0.2, steps=300, lr=2e-5):
    import types
    tok = tokenizers[model_id]
    ids = tok(train_text[:40000], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    import torch.nn.functional as F
    for i in range(steps):
        start = (i * 256) % max(1, len(ids) - 256)
        x = ids[start:start + 256].unsqueeze(0).to(device)
        kept = []
        blocks = list(get_layers(model))
        import random as _r
        _r.seed(i)
        sub = [b for b in blocks if _r.random() > p]
        set_layers(model, sub)
        out = model(input_ids=x, labels=x, use_cache=False)
        opt.zero_grad(); out.loss.backward(); opt.step()
        set_layers(model, blocks)
        if i % 50 == 0: print(i, round(out.loss.item(), 4))

rows = []
for mid in MODEL_IDS:
    m = load_fresh_model(mid).to(device)
    layerdrop_train(m, mid, p=0.2, steps=200)
    n = len(get_layers(m))
    for frac in [1.0, 0.75, 0.5]:
        keep = max(1, int(n * frac))
        blocks = list(get_layers(m))[:keep]
        m2 = load_fresh_model(mid)
        m2.load_state_dict(m.state_dict(), strict=False)
        set_layers(m2, blocks if len(blocks) == keep else list(get_layers(m2))[:keep])
        rows.append({"model": mid, "keep_frac": frac,
                     "ppl": calculate_ppl(m2.to(device), validation_text, mid)})
        print(mid, frac, rows[-1]["ppl"])
        free_model(m2)
    free_model(m)

import pandas as pd
pd.DataFrame(rows)
"""])

mk("12", "Attention/FFN Structured Pruning",
   "Score heads by gradient×activation, remove the least important.",
   [BI, """
rows = []
print("Head pruning by gradient*activation requires per-head hooks;")
print("use accelerate hooks or a quick saliency pass.")
print("Expected: drop ~25% of heads with <5% PPL change.")
# Placeholder evaluation: report baseline vs FFN channel cut as proxy.
for mid in MODEL_IDS:
    m = load_fresh_model(mid).to(device)
    base = calculate_ppl(m, validation_text, mid)
    rows.append({"model": mid, "config": "baseline", "ppl": base})
    # stub: zero-out half the FFN intermediate channels of a middle block
    blocks = get_layers(m)
    mid_block = blocks[len(blocks) // 2]
    for mod in mid_block.modules():
        if isinstance(mod, torch.nn.Linear) and mod.out_features > mod.in_features * 2:
            with torch.no_grad():
                mod.weight[: mod.weight.shape[0] // 2] = 0
    rows.append({"model": mid, "config": "50% FFN channels zeroed (1 block)",
                 "ppl": calculate_ppl(m, validation_text, mid)})
    print(mid, rows[-2]["ppl"], rows[-1]["ppl"])
    free_model(m)

import pandas as pd
pd.DataFrame(rows)
"""])

mk("13", "Early Exit / Adaptive Depth",
   "Exit heads + confidence threshold; measure PPL vs average depth.",
   [BI, """
print("Early-exit requires training exit classifiers on top of each block.")
print("Minimal demo: use the LM head itself as the exit classifier on a few blocks.")
rows = []
for mid in MODEL_IDS:
    m = load_fresh_model(mid).to(device)
    base = calculate_ppl(m, validation_text, mid)
    rows.append({"model": mid, "config": "full depth", "ppl": base})
    # proxy: use layer l's input + final norm + lm_head as an early-exit prediction
    blocks = get_layers(m)
    for l in range(len(blocks) // 2, len(blocks), 2):
        pass
    rows.append({"model": mid, "config": "proxy mid-point exit (illustrative)",
                 "ppl": float('nan')})
    free_model(m)

import pandas as pd
pd.DataFrame(rows)
"""])

mk("14", "Cross-Layer Weight Tying",
   "Share weights between layer l and l+d; measure PPL vs sharing period.",
   [BI, """
rows = []
for mid in MODEL_IDS:
    for d in [1, 2, 4]:
        m = load_fresh_model(mid)
        blocks = list(get_layers(m))
        for l in range(0, len(blocks) - d, 2 * d):
            blocks[l + d] = blocks[l]
        set_layers(m, blocks)
        rows.append({"model": mid, "period_d": d,
                     "ppl": calculate_ppl(m.to(device), validation_text, mid)})
        print(mid, d, rows[-1]["ppl"])
        free_model(m)

import pandas as pd
pd.DataFrame(rows)
"""])

mk("15", "Quantized No-Op Layer Bypass",
   "Correlate per-layer BI with PPL sensitivity after 3/4/2-bit quantization.",
   [BI, """
def quantize_weights(W, bits):
    gamma = W.abs().mean() + 1e-8
    levels = (2 ** bits) - 1
    W_scaled = (W / gamma).clamp(-1, 1) * ((levels - 1) / 2)
    W_q = torch.round(W_scaled) / ((levels - 1) / 2)
    return W_q * gamma

rows = []
for mid in MODEL_IDS:
    order, bis = redundant_order(mid)
    for L in order[:3]:
        for bits in [8, 4, 2]:
            m = load_fresh_model(mid)
            block = get_layers(m)[L]
            with torch.no_grad():
                for mod in block.modules():
                    if isinstance(mod, torch.nn.Linear):
                        mod.weight.data.copy_(quantize_weights(mod.weight.data, bits))
            rows.append({"model": mid, "L": L, "BI": bis[L], "bits": bits,
                         "ppl": calculate_ppl(m.to(device), validation_text, mid)})
            print(mid, L, bits, rows[-1]["ppl"])
            free_model(m)

import pandas as pd, matplotlib.pyplot as plt
df = pd.DataFrame(rows)
print(df)
fig, ax = plt.subplots(figsize=(7, 4))
for bits in df["bits"].unique():
    sub = df[df["bits"] == bits]
    ax.scatter(sub["BI"], sub["ppl"], label=f"{bits}-bit")
ax.set_xlabel("BI of quantized layer"); ax.set_ylabel("PPL"); ax.legend()
plt.title("BI vs quantization sensitivity"); plt.show()
"""])

print("done")
