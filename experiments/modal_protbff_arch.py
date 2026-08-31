"""Run the ProtBFF architecture sweep (antisym/dual/...) on Modal GPU.

Used to run the ESM2 ProtBFF arch on MVA-60 without waiting in the cluster
gpu queue. Bakes the score cache + arch code + MVA folds into the image;
writes the result JSON to a Volume and also returns it as text.

  modal run experiments/modal_protbff_arch.py            # ESM2, embed_dim 1280
"""
import modal

BASE = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF"
CACHE = f"{BASE}/model_benchmarking/score_caches/skempi_esm2_score_cache.npz"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "scikit-learn")
    .add_local_dir(f"{BASE}/tuning_v1", "/root/tuning_v1", copy=True)
    .add_local_dir(f"{BASE}/data/cross_validation_folds_mva/60_percent", "/root/folds", copy=True)
    .add_local_file(CACHE, "/root/cache.npz", copy=True)
)

app = modal.App("protbff-arch", image=image)
vol = modal.Volume.from_name("protbff-arch-out", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/root/out": vol}, timeout=6 * 3600)
def run(embed_dim: int = 1280, variants: str = "antisym,dual", seeds: int = 5,
        out: str = "arch_esm2.json"):
    import subprocess, sys, os
    cmd = [sys.executable, "/root/tuning_v1/protbff_arch.py",
           "--cache", "/root/cache.npz", "--folds_dir", "/root/folds",
           "--clusters", "/root/folds/clusters.tsv", "--variants", variants,
           "--embed_dim", str(embed_dim), "--seeds", str(seeds),
           "--out", f"/root/out/{out}"]
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    txt = open(f"/root/out/{out}").read() if os.path.exists(f"/root/out/{out}") else None
    return rc, txt


@app.local_entrypoint()
def main():
    rc, txt = run.remote(embed_dim=1280, variants="antisym,dual", seeds=5, out="arch_esm2.json")
    print("returncode", rc)
    if txt:
        open(f"{BASE}/tuning_v1/out/arch_esm2.json", "w").write(txt)
        print("saved arch_esm2.json locally")
