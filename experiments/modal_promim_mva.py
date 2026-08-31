"""Train ProMIM on the honest MVA-60 split, on Modal — PARALLEL by fold.

Serial (all 10 folds in one container) is ~20h at ~2h/fold on A10G and only saves
combined results at the very end, so it times out with nothing. Instead run each
fold in its own container via .map() → ~2h wall-clock, each fold saves its own CSV.

  modal run experiments/modal_promim_mva.py            # all 10 folds in parallel
  modal run experiments/modal_promim_mva.py::main --smoke   # 1 fold, 40 iters
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


@app.function(gpu="A10G", volumes={"/root/out": vol}, timeout=5 * 3600)
def train_fold(fold: int, max_iters: int = 5000):
    import os, re, subprocess, sys, glob
    os.chdir("/root/promim")
    base = open("configs/train/promim_ddg_skempi.yml").read()
    base = re.sub(r"max_iters:\s*\d+", f"max_iters: {max_iters}", base)
    base = re.sub(r"val_freq:\s*\d+", f"val_freq: {min(max_iters, 1000)}", base)
    cfg = f"configs/train/_modal_f{fold}.yml"
    open(cfg, "w").write(base)
    logdir = f"/root/out/fold_{fold}"
    cmd = [sys.executable, "train_promim_skempi.py", "--config", cfg,
           "--num_cvfolds", "10", "--fold_only", str(fold),
           "--logdir", logdir, "--tag", f"f{fold}", "--device", "cuda", "--num_workers", "4"]
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    hits = sorted(glob.glob(f"{logdir}/**/combined_all_folds_results.csv", recursive=True),
                  key=os.path.getmtime)
    csv = open(hits[-1]).read() if hits else None
    return fold, rc, csv


@app.local_entrypoint()
def main(smoke: bool = False):
    import os
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    import numpy as np, io, csv as _csv

    if smoke:
        print(train_fold.remote(0, max_iters=40)[:2]); return

    frames = {}
    for fold, rc, txt in train_fold.map(range(10), kwargs={"max_iters": 5000}):
        print(f"fold {fold}: rc={rc} rows={0 if not txt else txt.count(chr(10))-1}", flush=True)
        if txt:
            frames[fold] = txt

    # combine + metrics (mean-of-folds and pooled), save locally
    base = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/tuning_v1/out"
    os.makedirs(f"{base}/promim_folds", exist_ok=True)
    ally, allp, fr, fs, fa = [], [], [], [], []
    for fold in sorted(frames):
        open(f"{base}/promim_folds/fold_{fold}.csv", "w").write(frames[fold])
        rows = list(_csv.DictReader(io.StringIO(frames[fold])))
        y = np.array([float(r["ddG"]) for r in rows]); p = np.array([float(r["ddG_pred"]) for r in rows])
        fr.append(pearsonr(y, p)[0]); fs.append(spearmanr(y, p)[0])
        lab = (y > 0).astype(int)
        fa.append(roc_auc_score(lab, p) if len(set(lab)) > 1 else float("nan"))
        ally += list(y); allp += list(p)
    ally, allp = np.array(ally), np.array(allp)
    print(f"\n=== ProMIM MVA-60 ({len(frames)} folds, n={len(ally)}) ===")
    print(f"  mean-of-folds  P={np.nanmean(fr):.4f}  S={np.nanmean(fs):.4f}  AUROC={np.nanmean(fa):.4f}")
    print(f"  pooled         P={pearsonr(ally, allp)[0]:.4f}  S={spearmanr(ally, allp)[0]:.4f}")
