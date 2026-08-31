"""Extract RDE entropy features (RDE-Linear) on Modal, using SIM.pt as the frozen
rotamer density estimator. The linear calibration on MVA folds is done locally
afterwards from the returned pickle.

  modal run experiments/modal_rde_linear.py
"""
import modal

CODE = "/n/netscratch/shakhnovich_lab/Lab/jwang/rde_linear_mva"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy", "pandas", "scipy", "scikit-learn", "biopython",
                 "tqdm", "easydict", "pyyaml", "lmdb", "matplotlib")
    .add_local_dir(CODE, "/root/rde", copy=True)
)
app = modal.App("rde-linear", image=image)
vol = modal.Volume.from_name("rde-linear-out", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/root/out": vol}, timeout=4 * 3600)
def extract():
    import subprocess, sys, os
    os.chdir("/root/rde")
    # torch>=2.6 weights_only compat for the easydict-bearing SIM.pt
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
    cmd = [sys.executable, "-m", "rde.linear.entropy", "-c", "./trained_models/RDE.pt",
           "-o", "/root/out/entropy.pkl", "--device", "cuda"]
    print("RUN:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    vol.commit()
    return rc


@app.function(volumes={"/root/out": vol})
def fetch():
    import os
    p = "/root/out/entropy.pkl"
    return open(p, "rb").read() if os.path.exists(p) else None


@app.local_entrypoint()
def main():
    rc = extract.remote()
    print("extract rc:", rc)
    if rc == 0:
        data = fetch.remote()
        if data:
            open("/n/netscratch/shakhnovich_lab/Lab/jwang/rde_linear_mva/entropy.pkl", "wb").write(data)
            print(f"saved entropy.pkl ({len(data)} bytes)")
