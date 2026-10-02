"""
ShortGPT-style layer reduction via Block Influence (angular distance).

Usage:
    pip install torch transformers datasets accelerate
    python shortgpt_prune.py --model Qwen/Qwen2.5-0.5B --remove 8
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def get_block_inputs_outputs(model, tokenizer, texts, device):
    """Capture (input, output) hidden states of every decoder layer."""
    captured = {"inputs": [[] for _ in model.model.layers],
                "outputs": [[] for _ in model.model.layers]}
    hooks = []

    def make_hook(idx):
        def hook(module, args, output):
            x_in = args[0][0].detach().float()
            x_out = (output[0] if isinstance(output, tuple) else output).detach().float()
            # per-token angular distance, averaged over batch & sequence
            bi = (1 - F.cosine_similarity(x_in, x_out, dim=-1)).mean().item()
            captured["inputs"][idx].append(bi)  # store BI directly
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to(device)
            model(**enc)

    for h in hooks:
        h.remove()
    return captured


def block_influence(captured):
    """Angular distance: BI_l = 1 - cos(x_in, x_out). Small BI => near identity => safe to prune."""
    bis = []
    for bi_list in captured["inputs"]:
        bis.append(sum(bi_list) / len(bi_list))
    return bis


@torch.no_grad()
def perplexity(model, tokenizer, texts, device):
    model.eval()
    total_nll, total_tok = 0.0, 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model(**enc, labels=enc["input_ids"], use_cache=False)
        n = enc["input_ids"].numel()
        total_nll += out.loss.item() * n
        total_tok += n
    return torch.exp(torch.tensor(total_nll / total_tok)).item()


def prune_layers(model, indices_to_remove):
    keep = [l for i, l in enumerate(model.model.layers) if i not in indices_to_remove]
    model.model.layers = torch.nn.ModuleList(keep)
    # fix config so generate/save work
    model.config.num_hidden_layers = len(keep)
    if hasattr(model.config, "layer_types") and model.config.layer_types is not None:
        model.config.layer_types = [t for i, t in enumerate(model.config.layer_types)
                                    if i not in indices_to_remove]
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--remove", type=int, default=8, help="number of layers to prune")
    ap.add_argument("--calib_n", type=int, default=16)
    ap.add_argument("--save", default="pruned_model")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib = [t for t in ds["text"] if len(t.strip()) > 100][: args.calib_n]

    captured = get_block_inputs_outputs(model, tok, calib, device)
    bis = block_influence(captured)
    for i, bi in enumerate(bis):
        print(f"layer {i:2d}: BI = {bi:.5f}")

    # remove layers with SMALLEST BI (least influence)
    order = sorted(range(len(bis)), key=lambda i: bis[i])
    remove = sorted(order[: args.remove])
    print(f"\nRemoving layers: {remove}")

    ppl_before = perplexity(model, tok, calib[:8], device)
    print(f"PPL before: {ppl_before:.2f}")

    prune_layers(model, set(remove))
    ppl_after = perplexity(model, tok, calib[:8], device)
    print(f"PPL after : {ppl_after:.2f}")

    model.save_pretrained(args.save)
    tok.save_pretrained(args.save)
    print(f"Saved to ./{args.save}")


if __name__ == "__main__":
    main()
