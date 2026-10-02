import importlib.util
import torch

spec = importlib.util.spec_from_file_location("ps", "scripts/paper_study.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)
from transformers import AutoTokenizer

train, val, test = ps.load_texts()
mid = "openai-community/gpt2"
tok = AutoTokenizer.from_pretrained(mid)
order, _ = ps.redundant_order(mid, tok, train)
print("BI order:", order[:6])

m0 = ps.load_model(mid).to(ps.device)
L = order[0]
d = ps.hidden_size(m0)
X, Y = ps.collect_layer_io(m0, tok, L, train, n_chars=4000)
print("X", tuple(X.shape), "residual var", float(((Y - X) ** 2).mean()))

Xd = X.double()
D = (Y - X).double()
G = Xd.T @ Xd
G += 1e-3 * torch.eye(d, dtype=torch.float64) * G.diagonal().mean()
W = torch.linalg.solve(G, Xd.T @ D)
R = Xd @ W - D
ev_opt = 1 - float(R.pow(2).mean()) / float(D.pow(2).mean())
print(f"OPTIMAL full-rank linear EV = {ev_opt:.4f}")

U_, S_, Vt_ = torch.linalg.svd(W, full_matrices=False)
energy = (S_ ** 2).cumsum(0) / (S_ ** 2).sum()
for r in (1, 4, 16, 32, 64, 128, 256, 384):
    if r < len(energy):
        print(f"  rank {r:4d}: {float(energy[r-1]):.4f} of map energy")

# What does a *rank-r truncated* optimal map actually achieve on the residual?
print("\ntruncated-SVD fit of the residual map (upper bound for our bridge):")
Ur = U_[:, :32]
Sr = S_[:32]
Vr = Vt_[:32]
Wk = Ur @ torch.diag(Sr) @ Vr
for r in (4, 16, 32, 64, 128):
    Wk = Ur[:, :r] @ torch.diag(Sr[:r]) @ Vr[:r]
    Rk = Xd @ Wk - D
    ev = 1 - float(Rk.pow(2).mean()) / float(D.pow(2).mean())
    print(f"  rank {r:4d}: EV = {ev:.4f}")

ps.free(m0)