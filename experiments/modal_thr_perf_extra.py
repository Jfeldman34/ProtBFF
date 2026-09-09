"""Figure 2A/2B extra encoders: ESM2 / ESM3 / SaProt across MVA + CD-HIT thresholds.
ProtBFF (dual) + bare (ridge[D|S]); one GPU task per encoder, both splits per task.
Bakes the three caches + both fold trees.

  modal run experiments/modal_thr_perf_extra.py
"""
import modal

BASE = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF"
SC = f"{BASE}/model_benchmarking/score_caches"
ENC = {"esm2":   (f"{SC}/skempi_esm2_score_cache.npz", 1280),
       "esm3":   (f"{SC}/skempi_esm3_score_cache.npz", 1536),
       "saprot": (f"{SC}/skempi_saprot_score_cache.npz", 1280)}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "scikit-learn")
    .add_local_dir(f"{BASE}/tuning_v1", "/root/tuning_v1", copy=True,
                   ignore=["out/**", "out", "**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(f"{BASE}/data/cross_validation_folds_mva", "/root/mva", copy=True)
    .add_local_dir(f"{BASE}/data/cross_validation_folds_final", "/root/cdhit", copy=True)
    .add_local_file(ENC["esm2"][0], "/root/esm2.npz", copy=True)
    .add_local_file(ENC["esm3"][0], "/root/esm3.npz", copy=True)
    .add_local_file(ENC["saprot"][0], "/root/saprot.npz", copy=True)
)
app = modal.App("protbff-thrperf-extra", image=image)
vol = modal.Volume.from_name("protbff-thrperf-out", create_if_missing=True)

SPLITS = {"mva": ("/root/mva", "30,40,50,60,80,90"),
          "cdhit": ("/root/cdhit", "40,60,80,95,100")}


@app.function(gpu="A10G", volumes={"/root/out": vol}, timeout=6 * 3600)
def run(enc: str, embed_dim: int, seeds: int = 3):
    import subprocess, sys, os
    res = {}
    for split, (root, thr) in SPLITS.items():
        out = f"/root/out/thr_perf_{split}_{enc}.json"
        cmd = [sys.executable, "/root/tuning_v1/threshold_perf_sweep.py",
               "--cache", f"/root/{enc}.npz", "--embed_dim", str(embed_dim),
               "--folds_root", root, "--thresholds", thr, "--seeds", str(seeds), "--out", out]
        print(f"RUN {enc} {split}", flush=True)
        subprocess.run(cmd)
        res[split] = open(out).read() if os.path.exists(out) else None
    vol.commit()
    return enc, res


@app.local_entrypoint()
def main():
    args = [(e, ENC[e][1]) for e in ENC]
    for enc, res in run.starmap(args):
        for split, txt in res.items():
            if txt:
                open(f"{BASE}/tuning_v1/out/thr_perf_{split}_{enc}.json", "w").write(txt)
                print(f"saved thr_perf_{split}_{enc}.json")
