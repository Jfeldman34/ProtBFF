# Biophysical Score Calculation Pipeline

A parallelized pipeline for computing biophysical scores across FoldX-relaxed protein structures. Designed for high-throughput analysis of protein variants with checkpoint support for cluster computing.

## Table of Contents
- [Overview](#overview)
- [File Organization](#file-organization)
- [Prerequisites](#prerequisites)
- [Available Score Types](#available-score-types)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [SLURM Cluster Usage](#slurm-cluster-usage)
- [Command-Line Reference](#command-line-reference)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

## Overview

This pipeline computes per-residue biophysical features from protein structures that have been relaxed using FoldX. It supports five different score types and is optimized for parallel processing on HPC clusters.

**Important**: All PDB files must be FoldX-relaxed before score calculation.

## File Organization

```
your_project/
├── score_pipeline.py          # Main pipeline script
├── template_comparison.py     # OST comparison script (for lddt)
├── compute_all_scores.sh      # Bash script to compute all scores
├── scores/                     # Score calculation modules
│   ├── __init__.py
│   ├── burial_score.py        # k-nearest neighbor burial score
│   ├── interface_score.py     # Interface contact score
│   ├── sasa_score.py          # Solvent accessible surface area
│   ├── dihedral_score.py      # Dihedral angle changes
│   └── lddt_score.py          # Local distance difference test
├── data/
│   ├── mutations.csv          # Input CSV with index and ddG columns
│   ├── wildtype/              # FoldX-relaxed wildtype PDB files
│   ├── optimized/             # FoldX-relaxed mutant PDB files
│   └── ost_json/              # OST comparison JSON files (for lddt)
├── .checkpoints/              # Auto-created checkpoint files
├── logs/                       # Log files
└── requirements.txt
```

## Prerequisites

### 1. FoldX Relaxed Structures

**CRITICAL**: All PDB files must be relaxed using FoldX before score calculation.

- **Wildtype structures**: FoldX-relaxed reference structures (required for `dihedral` score)
- **Optimized/mutant structures**: FoldX-relaxed variant structures (required for all scores)

PDB files should follow the naming pattern: `{index}_*.pdb`

Examples:
```
0_mutant.pdb
1_mutant.pdb
2_variant.pdb
```

**Note**: Files containing "BLOOM" in the filename are automatically excluded.

### 2. Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.3.0
- numpy >= 1.20.0
- biopython >= 1.79
- tqdm >= 4.62.0

### 3. OpenStructure (OST) - Required for lDDT Scores

OpenStructure must be installed to compute lDDT scores via structural comparison.

**Installation options**:
- Via conda: `conda install bioconda::openstructure`
- From source: https://openstructure.org/
- Using Singularity container (see SLURM examples below)

**Note**: OST is only required if you plan to compute lDDT scores.

## Available Score Types

| Score | Description | Parameters | Requirements |
|-------|-------------|------------|--------------|
| **burial** | k-nearest neighbor burial | `--neighbors` (default: 9) | Optimized PDBs |
| **interface** | Interface contact score | `--sigma_interface` (default: 1.0) | Optimized PDBs |
| **sasa** | Solvent accessible surface area | `--sigma_sasa` (default: 1.0) | Optimized PDBs |
| **dihedral** | Backbone dihedral angle changes | None | Wildtype + Optimized PDBs |
| **lddt** | Local distance difference test | `--ost_json_dir` (required) | Optimized PDBs + OST JSONs |

## Quick Start

### 1. Edit Configuration

Edit `compute_all_scores.sh` and set your paths:

```bash
CSV_FILE="data/mutations.csv"
WILDTYPE_DIR="/path/to/foldx_relaxed/wildtype"    # For dihedral
OPTIMIZED_DIR="/path/to/foldx_relaxed/optimized"  # REQUIRED
TEMPLATE_PDB="/path/to/template.pdb"               # For lddt
```

### 2. Run All Scores

```bash
chmod +x compute_all_scores.sh
./compute_all_scores.sh
```

This will compute all scores sequentially: burial, interface, sasa, dihedral, and lddt.

## Usage Examples

### Basic Score Calculation

```bash
# Compute burial scores
python score_pipeline.py \
    --score burial \
    --csv data/mutations.csv \
    --pdb_opt_dir /path/to/foldx_relaxed/optimized/ \
    --ddg_column "ddG_LY-CoV555"

# Compute interface scores
python score_pipeline.py \
    --score interface \
    --csv data/mutations.csv \
    --pdb_opt_dir /path/to/foldx_relaxed/optimized/ \
    --sigma_interface 2.0

# Compute SASA scores
python score_pipeline.py \
    --score sasa \
    --csv data/mutations.csv \
    --pdb_opt_dir /path/to/foldx_relaxed/optimized/
```

### Dihedral Scores (Requires Wildtype)

```bash
python score_pipeline.py \
    --score dihedral \
    --csv data/mutations.csv \
    --pdb_wt_dir /path/to/foldx_relaxed/wildtype/ \
    --pdb_opt_dir /path/to/foldx_relaxed/optimized/
```

### lDDT Scores (Two-Step Process)

**Step 1**: Generate OST comparison files

```bash
python template_comparison.py \
    --template_pdb /path/to/template.pdb \
    --opt_dir /path/to/foldx_relaxed/optimized/ \
    --output_json_dir /path/to/ost_json/
```

**Step 2**: Compute lDDT scores

```bash
python score_pipeline.py \
    --score lddt \
    --csv data/mutations.csv \
    --pdb_opt_dir /path/to/foldx_relaxed/optimized/ \
    --ost_json_dir /path/to/ost_json/
```

### Using Custom ddG Column

```bash
# Use different antibody column
python score_pipeline.py \
    --score burial \
    --csv data/mutations.csv \
    --pdb_opt_dir /path/to/optimized/ \
    --ddg_column "ddG_ACE2"
```

### Parallel Processing

```bash
# Use 32 workers
python score_pipeline.py \
    --score burial \
    --csv data/mutations.csv \
    --pdb_opt_dir /path/to/optimized/ \
    --n_jobs 32
```

## SLURM Cluster Usage

### Example: SASA Score Calculation

```bash
#!/bin/bash
#SBATCH --job-name=sasa_score
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=your_partition
#SBATCH --mail-type=BEGIN,END,FAIL

# Change to working directory
cd /path/to/your/project/

# Load Conda environment
source /path/to/miniforge/etc/profile.d/conda.sh
conda activate your_env

# Run score calculation
python score_pipeline.py \
    --score sasa \
    --csv "/path/to/mutations.csv" \
    --pdb_opt_dir "/path/to/foldx_relaxed/optimized/" \
    --ddg_column "ddG_LY-CoV555" \
    --output "/path/to/output/sasa_scores/" \
    --n_jobs 32 \
    &> "/path/to/logs/sasa_${SLURM_JOB_ID}.log"
```

### Example: Dihedral Score Calculation

```bash
#!/bin/bash
#SBATCH --job-name=dihedral_score
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=your_partition
#SBATCH --mail-type=BEGIN,END,FAIL

cd /path/to/your/project/
source /path/to/miniforge/etc/profile.d/conda.sh
conda activate your_env

python score_pipeline.py \
    --score dihedral \
    --csv "/path/to/mutations.csv" \
    --pdb_wt_dir "/path/to/foldx_relaxed/wildtype/" \
    --pdb_opt_dir "/path/to/foldx_relaxed/optimized/" \
    --ddg_column "ddG_LY-CoV555" \
    --output "/path/to/output/dihedral_scores/" \
    --n_jobs 32 \
    &> "/path/to/logs/dihedral_${SLURM_JOB_ID}.log"
```

### Example: lDDT Calculation with OST (Using Singularity)

**Step 1**: Generate OST JSON files

```bash
#!/bin/bash
#SBATCH --job-name=lddt_ost
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=your_partition
#SBATCH --mail-type=BEGIN,END,FAIL

cd /path/to/your/project/

# Prepare log
LOG_DIR="/path/to/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/lddt_ost_$(date +'%Y-%m-%d_%H-%M-%S').log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== OST Comparison started at $(date) ==="
echo "Running on host: $(hostname)"

# Run with Singularity container for OST
singularity exec --bind /path/to/conda/env:/opt/ost ost.img bash -c '
    export PATH=/opt/ost/bin:$PATH
    export LD_LIBRARY_PATH=/opt/ost/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/opt/ost/lib/python3.11/site-packages:$PYTHONPATH
    
    python3 template_comparison.py \
        --template_pdb="/path/to/template.pdb" \
        --opt_dir="/path/to/foldx_relaxed/optimized/" \
        --output_json_dir="/path/to/ost_json/"
'

echo "=== Job finished at $(date) ==="
```

**Step 2**: Compute lDDT scores

```bash
#!/bin/bash
#SBATCH --job-name=lddt_score
#SBATCH --account=your_account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=your_partition
#SBATCH --mail-type=BEGIN,END,FAIL

cd /path/to/your/project/
source /path/to/miniforge/etc/profile.d/conda.sh
conda activate your_env

python score_pipeline.py \
    --score lddt \
    --csv "/path/to/mutations.csv" \
    --pdb_opt_dir "/path/to/foldx_relaxed/optimized/" \
    --ost_json_dir "/path/to/ost_json/" \
    --ddg_column "ddG_LY-CoV555" \
    --output "/path/to/output/lddt_scores/" \
    --n_jobs 32 \
    &> "/path/to/logs/lddt_${SLURM_JOB_ID}.log"
```

### Batch Job Submission

Save as `submit_all_scores.sh`:

```bash
#!/bin/bash

# Configuration
ACCOUNT="your_account"
PARTITION="your_partition"
PROJECT_DIR="/path/to/project"
CSV="/path/to/mutations.csv"
WT_DIR="/path/to/wildtype"
OPT_DIR="/path/to/optimized"
OUT_DIR="/path/to/output"
LOG_DIR="/path/to/logs"

mkdir -p "$LOG_DIR"

# Submit burial job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=burial
#SBATCH --account=$ACCOUNT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=$PARTITION

cd $PROJECT_DIR
source /path/to/conda.sh
conda activate your_env

python score_pipeline.py \
    --score burial \
    --csv "$CSV" \
    --pdb_opt_dir "$OPT_DIR" \
    --output "${OUT_DIR}/burial/" \
    --n_jobs 32 \
    &> "${LOG_DIR}/burial_\${SLURM_JOB_ID}.log"
EOF

# Submit interface job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=interface
#SBATCH --account=$ACCOUNT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=$PARTITION

cd $PROJECT_DIR
source /path/to/conda.sh
conda activate your_env

python score_pipeline.py \
    --score interface \
    --csv "$CSV" \
    --pdb_opt_dir "$OPT_DIR" \
    --output "${OUT_DIR}/interface/" \
    --n_jobs 32 \
    &> "${LOG_DIR}/interface_\${SLURM_JOB_ID}.log"
EOF

# Add similar blocks for sasa, dihedral, lddt...

echo "Jobs submitted. Check status with: squeue -u \$USER"
```

## Command-Line Reference

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--score` | str | Score type: burial, interface, sasa, dihedral, lddt |
| `--csv` | str | Input CSV file with index and ddG columns |
| `--pdb_opt_dir` | str | Directory with FoldX-relaxed mutant PDBs (REQUIRED) |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ddg_column` | str | `ddG_LY-CoV555` | Column name to filter by (non-NaN values) |
| `--pdb_wt_dir` | str | `data/wildtype` | Directory with wildtype PDBs (for dihedral) |
| `--ost_json_dir` | str | None | Directory with OST JSONs (for lddt) |
| `--output` | str | `{score}_scores` | Output file path (without extension) |
| `--n_jobs` | int | 32 | Number of parallel workers |
| `--neighbors` | int | 9 | k for burial score |
| `--sigma_interface` | float | 1.0 | σ for interface Gaussian weighting |
| `--sigma_sasa` | float | 1.0 | σ for SASA Gaussian weighting |
| `--csv_only` | flag | False | Only output CSV (skip NPZ) |
| `--npz_only` | flag | False | Only output NPZ (skip CSV) |
| `--checkpoint_freq` | int | 10 | Save checkpoint every N structures |
| `--no_checkpoint` | flag | False | Disable checkpointing |
| `--clear_checkpoint` | flag | False | Clear existing checkpoint |

## Output Format

### CSV Output
One row per residue in each structure:
```csv
index,chain,resnum,inscode,score
0,A,123, ,0.456
0,A,124, ,0.789
0,B,200, ,0.234
```

### NPZ Output
Compressed NumPy archive with dictionary structure:
- **Keys**: `(index, chain, resnum, inscode)` tuples
- **Values**: Score for that residue

Load with:
```python
import numpy as np
data = np.load('burial_scores.npz', allow_pickle=True)
score = data[str((0, 'A', 123, ' '))]
```

## Troubleshooting

### No results generated
- **Check PDB filenames**: Must match pattern `{index}_*.pdb`
- **Verify FoldX relaxation**: All PDBs must be FoldX-relaxed
- **Check index values**: CSV index column must match PDB filenames
- **Verify ddG column**: Ensure specified column has non-NaN values

### lddt score errors
- **Missing OST**: Install OpenStructure package
- **Missing JSON**: Run `template_comparison.py` first
- **JSON directory**: Verify `--ost_json_dir` path is correct
- **Filename mismatch**: JSON files must match PDB filenames

### Checkpoint issues
- **Clear stale checkpoints**: Use `--clear_checkpoint` flag
- **Check disk space**: Checkpoints stored in `.checkpoints/` directory
- **Configuration changes**: New parameters create new checkpoint files

### Memory errors on cluster
- **Reduce workers**: Lower `--n_jobs` value
- **Increase SLURM memory**: Adjust `#SBATCH --mem` parameter
- **Process in batches**: Split CSV into smaller chunks

## CSV File Format

Your input CSV must contain:
- **`index`** column: Integer identifier matching PDB filenames
- **ddG column**: Binding energy change (customizable name via `--ddg_column`)

Example CSV:
```csv
index,ddG_LY-CoV555,ddG_ACE2,mutation,chain,position
0,-0.5,-1.2,N501Y,A,501
1,0.3,-0.8,E484K,A,484
2,1.5,0.2,K417N,A,417
```

Rows with NaN in the ddG column are automatically filtered out.

## Performance Tips

1. **Parallel Processing**: Match `--n_jobs` to available CPUs
2. **Checkpoint Frequency**: Balance safety vs I/O (default: 10)
3. **Serial Debug Mode**: Use `--n_jobs 1` for troubleshooting
4. **PDB Caching**: Automatically caches parsed structures (maxsize=2000)
5. **Cluster Usage**: Submit scores as separate jobs for parallelism

## Citation

If you use this pipeline, please cite the relevant tools:

- **FoldX**: Schymkowitz et al. (2005) Nucleic Acids Research
- **OpenStructure**: Biasini et al. (2013) Journal of Structural Biology
- **Biopython**: Cock et al. (2009) Bioinformatics

## Support

For issues:
1. Check this README thoroughly
2. Verify FoldX relaxation of all structures
3. Review log files for specific error messages
4. Check file naming conventions and directory structure
5. Ensure OpenStructure is installed (for lddt)
