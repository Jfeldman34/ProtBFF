"""Train ProMIM on the honest MVA-60 split, on Modal (cluster env was purged).

Runtime deps are minimal (no dgl/atom3d/torch_geometric; unicore vendored as a
shim). Code + MVA folds are baked into the image; results go to a Volume.

  modal run experiments/modal_promim_mva.py --smoke   # 1-fold, 40 iters, verify it runs
  modal run experiments/modal_promim_mva.py           # full 10-fold, 5000 iters
  modal run experiments/modal_promim_mva.py::fetch     # print/download results
"""
import modal

CODE = "/n/netscratch/shakhnovich_lab/Lab/jwang/promim_mva"
FOLDS = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/data/cross_validation_folds_mva/60_percent"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "numpy", "pandas", "scipy", "scikit-learn", "biopython",
        "tqdm", "easydict", "pyyaml", "torchmetrics", "matplotlib", "lmdb",
        "joblib", "tensorboard",
    )
    .add_local_dir(CODE, "/root/promim", copy=True)
    .add_local_dir(FOLDS, "/root/mva_folds", copy=True)
)

app = modal.App("promim-mva", image=image)
vol = modal.Volume.from_name("promim-out", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/root/out": vol}, timeout=6 * 3600)
def train(num_cvfolds: int = 10, max_iters: int = 5000, tag: str = "full"):
    import os, re, subprocess, sys
    os.chdir("/root/promim")
    # build a config with the requested max_iters from the base skempi config
    base = open("configs/train/promim_ddg_skempi.yml").read()
    base = re.sub(r"max_iters:\s*\d+", f"max_iters: {max_iters}", base)
    base = re.sub(r"val_freq:\s*\d+", f"val_freq: {min(max_iters, 1000)}", base)
    cfg = f"configs/train/_modal_{tag}.yml"
    open(cfg, "w").write(base)
    cmd = [sys.executable, "train_promim_skempi.py", "--config", cfg,
           "--num_cvfolds", str(num_cvfolds), "--logdir", "/root/out",
           "--tag", tag, "--device", "cuda", "--num_workers", "4"]
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    print("returncode", rc, flush=True)
    return rc


@app.function(volumes={"/root/out": vol})
def fetch():
    """Return the newest combined results + overall metrics CSVs as text."""
    import os, glob
    out = {}
    for pat in ("**/combined_all_folds_results.csv", "**/overall_metrics.csv"):
        files = sorted(glob.glob(f"/root/out/{pat}", recursive=True), key=os.path.getmtime)
        if files:
            out[os.path.basename(files[-1])] = (files[-1], open(files[-1]).read())
    return out


@app.local_entrypoint()
def main(smoke: bool = False):
    if smoke:
        print("SMOKE:", train.remote(num_cvfolds=10, max_iters=40, tag="smoke"))
    else:
        print("FULL:", train.remote(num_cvfolds=10, max_iters=5000, tag="full"))
