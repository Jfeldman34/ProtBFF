"""Figure 2A recreation on the NEW (MVA all-versus-all) split method.

Runs ProtBFF (dual readout) for ESM-C and ProSST across the six MVA identity thresholds
(30/40/50/60/80/90%), in parallel on Modal, to show the leakage-driven decline in
performance as homology control tightens. Bakes the two caches + all six threshold fold
dirs into the image; each (encoder, threshold) is one GPU task.

  modal run experiments/modal_threshold_sweep.py
"""
import modal

BASE = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF"
ENCODERS = {"esmc": (f"{BASE}/model_benchmarking/score_caches/skempi_esmc_score_cache.npz", 1152),
            "prosst": (f"{BASE}/model_benchmarking/score_caches/skempi_score_cache.npz", 768)}
THRESHOLDS = [30, 40, 50, 60, 80, 90]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "scikit-learn")
    .add_local_dir(f"{BASE}/tuning_v1", "/root/tuning_v1", copy=True)
    .add_local_dir(f"{BASE}/data/cross_validation_folds_mva", "/root/mva", copy=True)
    .add_local_file(ENCODERS["esmc"][0], "/root/esmc.npz", copy=True)
    .add_local_file(ENCODERS["prosst"][0], "/root/prosst.npz", copy=True)
)
app = modal.App("protbff-thresh", image=image)
vol = modal.Volume.from_name("protbff-thresh-out", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/root/out": vol}, timeout=3 * 3600)
def run_one(encoder: str, embed_dim: int, thr: int, seeds: int = 3):
    import subprocess, sys, json, os
    cache = f"/root/{encoder}.npz"
    folds = f"/root/mva/{thr}_percent"
    out = f"/root/out/{encoder}_{thr}.json"
    cmd = [sys.executable, "/root/tuning_v1/protbff_arch.py",
           "--cache", cache, "--folds_dir", folds, "--clusters", f"{folds}/clusters.tsv",
           "--variants", "antisym,dual", "--embed_dim", str(embed_dim), "--seeds", str(seeds), "--out", out]
    print("RUN:", encoder, thr, flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    r = json.load(open(out)) if os.path.exists(out) else None
    d = r.get("dual", {}) if r else {}
    return dict(encoder=encoder, thr=thr, rc=rc,
                mean_r=d.get("mean_r"), mean_s=d.get("mean_s"),
                pooled_r=d.get("pooled_r"), pooled_s=d.get("pooled_s"))


@app.local_entrypoint()
def main():
    import json
    jobs = [(e, ENCODERS[e][1], t) for e in ENCODERS for t in THRESHOLDS]
    results = list(run_one.starmap(jobs))
    out = {}
    for r in results:
        out.setdefault(r["encoder"], {})[r["thr"]] = r
        print(f"{r['encoder']:7s} {r['thr']:3d}%  meanP={r['mean_r']}  meanS={r['mean_s']}")
    open(f"{BASE}/tuning_v1/out/threshold_sweep_mva.json", "w").write(json.dumps(out, indent=2))
    print("saved threshold_sweep_mva.json")
