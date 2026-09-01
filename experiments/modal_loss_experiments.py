"""Run the ProtBFF loss-function sweep on Modal GPU (no cluster queue wait).

Dual readout, ProSST cache, honest MVA-60, clean protocol, 5 seeds. Variants:
mse | huber | mae | pearson(0.5,1.0) | corr_only | rank | ilddt2 (antisym vs dual head).
Bakes cache + code + folds into the image; writes result JSON to a Volume and returns it.

  modal run experiments/modal_loss_experiments.py
"""
import modal

BASE = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF"
CACHE = f"{BASE}/model_benchmarking/score_caches/skempi_esmc_score_cache.npz"  # ESM-C (SOTA), 1152-d

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "scikit-learn")
    .add_local_dir(f"{BASE}/tuning_v1", "/root/tuning_v1", copy=True)
    .add_local_dir(f"{BASE}/data/cross_validation_folds_mva/60_percent", "/root/folds", copy=True)
    .add_local_file(CACHE, "/root/cache.npz", copy=True)
)

app = modal.App("protbff-loss", image=image)
vol = modal.Volume.from_name("protbff-loss-out", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/root/out": vol}, timeout=6 * 3600)
def run(embed_dim: int = 1152, seeds: int = 5, out: str = "loss_esmc.json"):
    import subprocess, sys, os
    cmd = [sys.executable, "/root/tuning_v1/loss_experiments.py",
           "--cache", "/root/cache.npz", "--folds_dir", "/root/folds",
           "--clusters", "/root/folds/clusters.tsv",
           "--embed_dim", str(embed_dim), "--seeds", str(seeds),
           "--out", f"/root/out/{out}"]
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    txt = open(f"/root/out/{out}").read() if os.path.exists(f"/root/out/{out}") else None
    return rc, txt


@app.local_entrypoint()
def main():
    rc, txt = run.remote(embed_dim=1152, seeds=5, out="loss_esmc.json")
    print("returncode", rc)
    if txt:
        open(f"{BASE}/tuning_v1/out/loss_esmc.json", "w").write(txt)
        print("saved loss_esmc.json locally")
