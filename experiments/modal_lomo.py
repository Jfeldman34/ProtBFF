"""Figure 2C recreation on Modal: leave-one-experimental-method-out, ESM-C + ProtBFF (dual).

Bakes the ESM-C cache + master folds CSV + raw SKEMPI2 (for Method labels) + code into
the image; runs tuning_v1/lomo.py on one GPU and returns the result JSON.

  modal run experiments/modal_lomo.py
"""
import modal

BASE = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF"
CACHE = f"{BASE}/model_benchmarking/score_caches/skempi_esmc_score_cache.npz"
MASTER = f"{BASE}/data/cross_validation_folds_mva/60_percent/folds_60pct.csv"
RAW = "/n/netscratch/shakhnovich_lab/Lab/jwang/rde_linear_mva/data/SKEMPI_v2/skempi_v2.csv"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "scikit-learn", "pandas")
    .add_local_dir(f"{BASE}/tuning_v1", "/root/tuning_v1", copy=True,
                   ignore=["out/**", "out", "**/__pycache__/**", "**/*.pyc"])
    .add_local_file(CACHE, "/root/esmc.npz", copy=True)
    .add_local_file(MASTER, "/root/master.csv", copy=True)
    .add_local_file(RAW, "/root/skempi_v2.csv", copy=True)
)
app = modal.App("protbff-lomo", image=image)
vol = modal.Volume.from_name("protbff-lomo-out", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/root/out": vol}, timeout=4 * 3600)
def run(embed_dim: int = 1152, seeds: int = 2, out: str = "lomo_esmc.json"):
    import subprocess, sys, os
    cmd = [sys.executable, "/root/tuning_v1/lomo.py", "--cache", "/root/esmc.npz",
           "--embed_dim", str(embed_dim), "--master", "/root/master.csv",
           "--raw", "/root/skempi_v2.csv", "--seeds", str(seeds), "--out", f"/root/out/{out}"]
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    txt = open(f"/root/out/{out}").read() if os.path.exists(f"/root/out/{out}") else None
    return rc, txt


@app.local_entrypoint()
def main():
    rc, txt = run.remote(embed_dim=1152, seeds=2, out="lomo_esmc.json")
    print("returncode", rc)
    if txt:
        open(f"{BASE}/tuning_v1/out/lomo_esmc.json", "w").write(txt)
        print("saved lomo_esmc.json locally")
