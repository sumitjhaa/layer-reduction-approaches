"""Patch notebooks: MPS device, per-notebook small dataset, lighter budgets."""
import nbformat, glob, json

DATASETS = {
    "01": ("Salesforce/wikitext", "wikitext-2-raw-v1", "text"),
    "02": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
    "03": ("Salesforce/wikitext", "wikitext-2-v1", "text"),
    "04": ("Salesforce/wikitext", "wikitext-2-raw-v1", "text"),
    "05": ("Salesforce/wikitext", "wikitext-103-v1", "text"),
    "06": ("Salesforce/wikitext", "wikitext-2-raw-v1", "text"),
    "07": ("Salesforce/wikitext", "wikitext-2-v1", "text"),
    "08": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
    "09": ("Salesforce/wikitext", "wikitext-103-v1", "text"),
    "10": ("Salesforce/wikitext", "wikitext-2-raw-v1", "text"),
    "11": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
    "12": ("Salesforce/wikitext", "wikitext-2-v1", "text"),
    "13": ("Salesforce/wikitext", "wikitext-103-v1", "text"),
    "14": ("Salesforce/wikitext", "wikitext-2-raw-v1", "text"),
    "15": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
}

for f in sorted(glob.glob("approaches/notebooks/*.ipynb")):
    nb = nbformat.read(f, as_version=4)
    num = f.split("/")[-1][:2]
    ds, cfg, field = DATASETS[num]
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        # 1) device -> mps
        if 'device = "cuda" if torch.cuda.is_available() else "cpu"' in src:
            src = src.replace(
                'device = "cuda" if torch.cuda.is_available() else "cpu"',
                'device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")',
            )
        # 2) dataset loader
        if 'dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")' in src:
            loader = f'dataset = load_dataset("{ds}"' + (f', "{cfg}"' if cfg else "") + ")"
            src = src.replace(
                'dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")', loader)
            # field name fix
            src = src.replace('x for x in split["text"]', f'x for x in split["{field}"]')
            # attach sample limiting
            if "train_text = join_nonempty" in src:
                src = src.replace("train_text = join_nonempty(dataset[\"train\"])",
                                  "train_text = join_nonempty(dataset[\"train\"])[:300000]")
                src = src.replace("validation_text = join_nonempty(dataset[\"validation\"])",
                                  "validation_text = join_nonempty(dataset[\"validation\"])[:100000]")
                src = src.replace("test_text = join_nonempty(dataset[\"test\"])",
                                  "test_text = join_nonempty(dataset[\"test\"])[:100000]")
        # 3) smaller eval tokens
        src = src.replace("EVAL_TOKENS = 5000", "EVAL_TOKENS = 3000")
        # 4) lighter training budgets
        src = src.replace("iters=200", "iters=80")
        src = src.replace("steps=300", "steps=60")
        src = src.replace("steps=200", "steps=60")
        src = src.replace("steps=100", "steps=60")
        cell.source = src
    nbformat.write(nb, f)
    print("patched", f)
