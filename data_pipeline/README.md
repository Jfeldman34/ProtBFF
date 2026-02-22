# ProtBFF Data Pipeline

End-to-end pipeline from FoldX-relaxed PDB structures to preprocessed feature vectors ready for ML training.

**CRITICAL**: All PDB files must be FoldX-relaxed before running any step.

---

## Overview

1. **Tokenize & Embed** — Generate per-residue ProSST embeddings from structures
2. **Merge Scores** — Compute biophysical scores and merge with embeddings
3. **Preprocess** — Apply max pooling to produce fixed-size feature vectors

---

## Step 1: ProSST Embeddings

If not there, place FoldX-relaxed PDB files in `ProtBFF/data/optimized/` and `ProtBFF/data/wildtype/`.

### Tokenize structures
```bash
python ProtBFF/data_pipeline/tokenize_pdb_full.py \
  --pdb-dir ProtBFF/data/optimized \
  --output-dir ProtBFF/data/optimized_tokens_2048

python ProtBFF/data_pipeline/tokenize_pdb_full.py \
  --pdb-dir ProtBFF/data/wildtype \
  --output-dir ProtBFF/data/wildtype_tokens_2048
```

### Generate embeddings
```bash
python ProtBFF/data_pipeline/embedding_pdb_full.py \
  --token-dir ProtBFF/data/optimized_tokens_2048 \
  --pdb-dir ProtBFF/data/optimized \
  --output-dir ProtBFF/data/optimized_embeddings_2048

python ProtBFF/data_pipeline/embedding_pdb_full.py \
  --token-dir ProtBFF/data/wildtype_tokens_2048 \
  --pdb-dir ProtBFF/data/wildtype \
  --output-dir ProtBFF/data/wildtype_embeddings_2048
```

Outputs `.fasta` token files and `*_embeddings.npy` files in their respective directories. Use `--overwrite` to recompute existing outputs.

---

## Step 2: Compute Scores & Merge (`calculate_all_scores.py`)

Computes all five biophysical scores (burial, interface, SASA, dihedral, lDDT) per residue and merges them with the embedding differences into a single NPZ file per structure.
```bash
python ProtBFF/data_pipeline/calculate_all_scores.py \
    --csv ProtBFF/data/SKEMPI2_filtered_final.csv \
    --pdb_wt_dir ProtBFF/data/wildtype \
    --pdb_opt_dir ProtBFF/data/optimized \
    --wt_embedding_dir ProtBFF/data/wildtype_embeddings_2048 \
    --opt_embedding_dir ProtBFF/data/optimized_embeddings_2048 \
    --output_dir ProtBFF/data_pipeline/merged_output \
    --temp_dir ProtBFF/data_pipeline/.temp_scores
```

For lDDT scores, also pass `--ost_json_dir` pointing to your OST JSON files (requires OpenStructure: `conda install bioconda::openstructure`).

### What it does internally

The script runs three steps:

1. **Score calculation** — For each structure index in the CSV, finds the matching PDB file (`{index}_*.pdb`), computes all five scores per residue, and saves intermediate CSVs to `--temp_dir`.
2. **NPZ extraction** — Converts each score CSV into per-structure NPZ files. Dihedral scores are min-max normalized per structure.
3. **Merge** — Loads all score arrays and the corresponding `.npy` embeddings, computes `Xf = wildtype − optimized` and `Xr = optimized − wildtype`, and saves everything into a single `merged_{index}.npz`.

### Output NPZ contents

| Key | Shape | Description |
|-----|-------|-------------|
| `burial` | `(L,)` | Per-residue burial scores |
| `interface` | `(L,)` | Per-residue interface scores |
| `sasa` | `(L,)` | Per-residue SASA scores |
| `dihedral` | `(L,)` | Per-residue dihedral scores (normalized) |
| `lddt` | `(L,)` | Per-residue lDDT scores |
| `Xf` | `(L, D)` | Wildtype − optimized embedding differences |
| `Xr` | `(L, D)` | Optimized − wildtype embedding differences |
| `ddG` | `(1,)` | Binding affinity change from CSV |
| `ilddt` | `(1,)` | Global interface lDDT (NaN if OST not used) |
| `index` | `(1,)` | Numeric structure index |
| `protein_id` | str | Protein identifier from `#Pdb` column in CSV |

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | — | Input CSV with `index` and ddG columns |
| `--pdb_wt_dir` | Yes | — | Wildtype PDB directory |
| `--pdb_opt_dir` | Yes | — | Optimized PDB directory |
| `--output_dir` | Yes | — | Output directory for merged NPZ files |
| `--ddg_column` | No | `ddG_LY-CoV555` | CSV column to filter on (rows with NaN are skipped) |
| `--id_column` | No | `#Pdb` | CSV column to use as protein identifier |
| `--ost_json_dir` | No | None | OST JSON directory (required for lDDT) |
| `--wt_embedding_dir` | No | None | Wildtype embeddings directory |
| `--opt_embedding_dir` | No | None | Optimized embeddings directory |
| `--temp_dir` | No | `.temp_scores` | Directory for intermediate score files |
| `--skip_calculation` | No | False | Skip score computation, use existing CSVs in `--temp_dir` |
| `--skip_extraction` | No | False | Skip NPZ extraction, use existing NPZ files in `--temp_dir` |

### CSV format
```csv
index,#Pdb,ddG_LY-CoV555,mutation,chain,position
0,1CSE,-0.5,N501Y,A,501
1,1PPF,0.3,E484K,A,484
```

---

## Step 3: Preprocess for ML Training (`merge_scores.py`)

Applies score-weighted max pooling over the embedding differences to produce one fixed-size feature vector per structure.
```bash
python ProtBFF/data_pipeline/merge_scores.py \
    --merged_dir ProtBFF/data_pipeline/merged_output \
    --output_cache ProtBFF/data_pipeline/preprocessed_data.npz
```

For each structure, each score type independently weights the `(L, D)` embedding difference matrix and takes the per-dimension maximum, yielding a `(D,)` vector. The five resulting vectors (interface, burial, lDDT, SASA, dihedral) are concatenated in that order, producing forward (`Xf`) and reverse (`Xr`) feature matrices of shape `(N, 5D)`.

Note that lDDT is inverted (`1 - lddt`) before pooling, since lower lDDT indicates greater structural deviation. Any NaN values in per-residue lDDT are imputed with the per-structure mean before pooling.

### Output cache contents

| Key | Shape | Description |
|-----|-------|-------------|
| `Xf` | `(N, 5D)` | Forward pooled features |
| `Xr` | `(N, 5D)` | Reverse pooled features |
| `y` | `(N,)` | ddG target values |
| `ilddt` | `(N,)` | Global interface lDDT values |
| `ids` | `(N,)` | Full protein identifiers (e.g. `"0_1CSE"`) |

On load, `ids` is split on `_` to recover short protein codes (e.g. `"1CSE"`) for cross-validation fold matching. If the cache file already exists at `--output_cache`, it is loaded directly without reprocessing. Use `--no_cache` to skip saving.