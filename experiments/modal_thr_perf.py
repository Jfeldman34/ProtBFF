"""Figure 2A on Modal: performance vs MVA sequence-identity threshold.

ProtBFF (dual) + bare (ridge[D|S]) for ESM-C and ProSST across MVA thresholds 30-90%.
Bakes the two caches + all MVA threshold fold dirs; one GPU task per encoder.

  modal run experiments/modal_thr_perf.py
"""
import modal

BASE = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF"
ENC = {"esmc": (f"{BASE}/model_benchmarking/score_caches/skempi_esmc_score_cache.npz", 1152),
       "prosst": (f"{BASE}/model_benchmarking/score_caches/skempi_score_cache.npz", 768)}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "scikit-learn")
    .add_local_dir(f"{BASE}/tuning_v1", "/root/tuning_v1", copy=True,
                   ignore=["out/**", "out", "**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(f"{BASE}/data/cross_validation_folds_mva", "/root/mva", copy=True)
    .add_local_file(ENC["esmc"][0], "/root/esmc.npz", copy=True)
    .add_local_file(ENC["prosst"][0], "/root/prosst.npz", copy=True)
)
app = modal.App("protbff-thrperf", image=image)
vol = modal.Volume.from_name("protbff-thrperf-out", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/root/out": vol}, timeout=6 * 3600)
def run(enc: str, embed_dim: int, seeds: int = 3):
    import subprocess, sys, os, json
    out = f"/root/out/thr_perf_mva_{enc}.json"
    cmd = [sys.executable, "/root/tuning_v1/threshold_perf_sweep.py",
           "--cache", f"/root/{enc}.npz", "--embed_dim", str(embed_dim),
           "--folds_root", "/root/mva", "--thresholds", "30,40,50,60,80,90",
           "--seeds", str(seeds), "--out", out]
    print("RUN:", enc, flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    return enc, rc, (open(out).read() if os.path.exists(out) else None)


@app.local_entrypoint()
def main():
    args = [(e, ENC[e][1]) for e in ENC]
    for enc, rc, txt in run.starmap(args):
        print(f"{enc}: rc={rc}")
        if txt:
            open(f"{BASE}/tuning_v1/out/thr_perf_mva_{enc}.json", "w").write(txt)
            print(f"  saved thr_perf_mva_{enc}.json")
