# ProtBFF — Project Notes

Upstream: https://github.com/Jfeldman34/ProtBFF (branch `main`)
Local: `/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF`
Paper: PNAS submission — injecting biophysical priors into PLM embeddings for ddG prediction

## 2026-08-09 — netscratch purge + re-clone (READ THIS FIRST)

The `/n/netscratch` purge wiped this checkout: 208 empty directories, **0 files**, including an
empty `.git` (so `git status` failed with "not a git repository"). Nothing was recoverable
locally — no uncommitted work survived.

Recovered by removing the empty dir tree and re-cloning from GitHub at
`9981012b Update requirements.txt`. Working tree is clean and matches `origin/main`:
21,213 files, 13,914 git-lfs objects (5.81 GiB) materialized — LFS content verified real,
not pointer stubs.

**Lost for good** (present pre-purge, not tracked upstream — must be regenerated):
- `analysis/`, `benchmarking/`, `paper/`, `scores/`, `slurm_logs/` (top-level, local-only)
- the previous `CLAUDE.md` (this file is a reconstruction, not the original)

**Note**: `git clone` here takes >2 min because of the LFS smudge — run it with
`run_in_background: true` or a long timeout, or the tool call dies mid-checkout and leaves
a partial working tree (fix with `git reset --hard origin/main`).

## Repo structure (as of 9981012b)

```
data/                        SKEMPI2_filtered_final.csv, cross_validation_folds_final/,
                             lddt_dir/, optimized/, wildtype/
data_pipeline/               calculate_all_scores.py, embedding_pdb_full.py,
                             tokenize_pdb_full.py, merge_scores.py, prosst/, scores/
model_benchmarking/          7kmg_/7w9i_/9lyp_comparison_output, score_caches,
                             skempi_prosst_output
antibodies_protbff_benchmarking.py
evaluate_saved_models.py
requirements.txt
```

git-lfs tracks `*.npz *.pt *.pth *.pkl *.h5 *.pdb *.bin *.zip` — git-lfs 3.4.1 is at
`/usr/bin/git-lfs` on the cluster.

## How the sequence split was done (investigated 2026-08-09)

**Dataset**: `data/SKEMPI2_filtered_final.csv` — 6,631 mutation rows over 335 complexes,
all `Label == forward`. `#Pdb` = `<row_idx>_<PDBCODE>`; rows/complex are very skewed
(median 6, max 295).

**Scheme**: grouped 10-fold CV at the *complex* level, in 7 variants:
`data/cross_validation_folds_final/{40,45,60,80,95,99,100}_percent/fold_{1..10}/`
with `train_complex_ids.txt` / `test_complex_ids.txt`. Verified for every variant:
test folds exactly partition the 335 complexes, and no complex appears in both train and
test of the same fold. `100_percent` = no clustering (33–34 complexes/fold, near-uniform);
the others cluster first, so fold sizes are lumpy (16–62 complexes).

**Provenance** — recovered from `args` inside
`model_benchmarking/skempi_prosst_output/fold_0_full_model.pth`:
```
grouping_csv: .../amaechler/DDAffinity_up/data/complex_sequences_cdhit_grouped_60.csv   [PURGED]
folds_dir:    .../jonathanfeldman/AF3Complex/cross_validation_folds_final/60_percent/
random_state: 42,  n_folds: 10,  epochs: 100,  lr: 1e-4,  batch_size: 32
```
So: **CD-HIT clustering of complex sequences, percentages = identity thresholds**, groups
then assigned to 10 folds. The clustering came from the DDAffinity pipeline.
**The released SKEMPI models are the 60% split** (checkpoint `fold_0` test_idx has 1,374
rows = `60_percent/fold_1`). The fold-generation script is NOT in this repo.

Surviving clustering inputs (read-only, another user's dir):
`/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/AF3Complex/clustering_csvs/complex_sequences_grouped_{40,60,80,95,99}.csv`
— columns `complex_name, chain_name, sequence, sequence_length, chain_group_id, complex_group_id`
(340 complexes; chains clustered, then complexes grouped transitively via shared chain groups).
No CSV exists for 45%, so that variant's provenance is unclear.

### Caveat: the thresholds are directional, not strict

Measured directly from wildtype PDB sequences (335 complexes, 831 chains, local alignment
+ exact-match check). Test rows whose complex shares a **byte-identical chain** with a
train complex in the same fold:

| split | 40% | 45% | 60% | 80% | 95% | 99% | 100% |
|---|---|---|---|---|---|---|---|
| test rows affected | 6.5% | 6.2% | **14.6%** | 16.2% | 32.9% | 34.9% | 41.3% |
| complexes affected | 26 | 23 | **54** | 59 | 78 | 87 | 145 |

Monotonic, so the thresholds genuinely do work — but none reaches zero. Concrete case: at
40%, `3HFM` is in fold_9 **test** while `1VFB`, `1MLC`, `1DQJ`, `1XGU` are in fold_9
**train**, and all five share the *identical* 129-residue hen lysozyme chain (verified by
string equality) — same antigen, different antibodies. The clustering CSVs *do* put all
five in `complex_group_id == 8`.

**The split is leaky, not broken.** Against a random-complex-split null (200 shuffles,
matched fold sizes), the folds respect the CD-HIT groups far better than chance:

| threshold | groups split across folds | random null | z |
|---|---|---|---|
| 40% | 18 | 39.3 ± 1.2 | −18.0 |
| 60% | 20 | 47.6 ± 1.4 | −20.1 |
| 80% | 31 | 53.3 ± 1.6 | −13.7 |
| 95% | 33 | 52.6 ± 1.8 | −10.9 |
| 99% | 35 | 44.1 ± 1.7 | −5.4 |

P(two complexes land in the same test fold | measured chain identity ≥ 90%), chance = 0.10:
0.83 at the 40% split, 0.77 at 60%, 0.34 at 99%, **0.08 at 100%** (i.e. no clustering, as
designed). Clustering clearly drove fold assignment; the residual is the problem, not the
mechanism.

No candidate grouping reproduces the folds exactly — transitive `complex_group_id`,
longest-chain group, shortest-chain group, first-chain group, and min-chain-group all leave
14–35 groups split at every threshold. The grouping CSV actually used at training time
(amaechler's) was purged, so the exact mechanism can't be pinned down; the observable defect
below is certain, its cause is inferred.

### Where the defect actually lives (traced 2026-08-09)

**The fold splitter is correct.** `create_cluster_folds()` in
`/n/netscratch/.../jonathanfeldman/AF3Complex/prosst_antoine.py:187` runs
`KFold(n_splits=10, shuffle=True, random_state=42)` over `np.unique(complex_groups)` and then
masks complexes by group membership — that construction *cannot* split a group across folds.
So the leak does not come from the splitting step.

**The generator is
`/n/netscratch/.../jonathanfeldman/DDAffinity_up-master/compute_sequence_clusters_cdhit.py`**
(its output filename pattern `complex_sequences_cdhit_grouped_{thr}.csv` matches the
`grouping_csv` path recorded in the released checkpoints exactly). What it does:

- `prepare_fasta_for_complexes()` concatenates a complex's chains into **one** FASTA record,
  joined by a literal `"X"` separator (line 63).
- `run_cdhit_clustering()` runs `cd-hit -i <fasta> -c <threshold> -n 2 -M 16000 -d 0`.
- `df['complex_group_id'] = df['complex_name'].map(complex_cluster_mapping)` — the cluster id
  is used **directly**.

**There is no merge step at all.** An earlier session note here claimed the defect was an
"incomplete union-find over shared chain clusters that never reached a fixpoint" — that was
wrong; no such step exists. `complex_group_id` is simply CD-HIT run on chain-concatenated
complex sequences.

**The real defect is that construction itself.** Clustering concatenated multi-chain sequences
does not yield a meaningful identity guarantee, because chain order and chain lengths differ
between complexes, so the alignment CD-HIT computes does not correspond to a biological one.
Both failure directions are present in the output:

- *Too permissive.* Group 0 at the 60% threshold holds 22 complexes (305–451 aa). Its longest
  member `3BX1` (451 aa, the CD-HIT representative, since representatives are the longest
  sequence) is **21.3–51.8%** identical to the other 21 — **0 of 21 reach 60%**. A valid 60%
  CD-HIT cluster cannot look like this. 12 of the 31 groups with ≥3 members have no member
  reaching 60% to all others.
- *Too strict where it matters.* Complexes sharing a byte-identical chain land in different
  clusters — `1AHW` (group 11) vs `1DQJ`/`1MLC`/`1XGP`/`1XGQ`/`1XGR`/`1XGT` (group 8) share
  chain cluster 25; 834 such pairs at the 60% threshold, spanning 142 of 340 complexes. This
  is the pathway that puts identical sequences on both sides of the split.

Also suspect: **`-n 2` is hardcoded for every threshold** (line 93, commented "Standard word
length for low thresholds"). CD-HIT's recommended word length is 3 for 0.5–0.6, 4 for
0.6–0.7, 5 for 0.7–1.0. Unverified here — cd-hit is not installed on the cluster — but it is
outside the documented range for the 60/80/95/99% runs.

Note the folds still match no surviving grouping CSV exactly, so the specific CSV consumed at
fold-generation time (amaechler's copy) remains purged; the script above is what produced it.

Also note: the repo's folds match **no** surviving grouping CSV exactly (12–20 groups broken
at the 40%/60% thresholds against every candidate), so the CSV actually used
(amaechler's `complex_sequences_cdhit_grouped_60.csv`) is purged and unrecoverable.

Consequence: a split labelled "40% sequence identity" does not guarantee that train and test
are below 40% identity. It reduces cross-split similarity substantially but does not bound
it. The gap between the label and the guarantee is the thing to fix or to state explicitly
in the paper.

**Relevance**: the paper's headline claim is about interface prediction, and shared-antigen
leakage inflates exactly the kind of held-out performance that claim rests on. Worth
re-running the 60% benchmark under a strict group split before submission.

## tm50 structural split (commit `34e1a8e1`, 2026-08-09)

New `tm50/` folder: 10 folds + `folds_tm50.csv` (adds a 0-indexed `fold` column to the SKEMPI
table) + `leak_audit/`. Fold dirs match the CSV's `fold` column exactly. No generation script
and no README entry shipped with it.

**Leakage: fixed, and independently verified.** Measured with my own alignments (not their
audit):

| | tm50 | 60_percent (old) |
|---|---|---|
| max train–test chain identity | **30.6%** | 100.0% |
| train–test pairs ≥90% id | **0** | 616 |
| train–test pairs ≥40% id | **0** | 3,490 |
| test rows sharing a byte-identical chain with train | **0 / 6631** | 971 (14.6%) |

Test folds partition the 335 complexes; no complex is in both train and test. This is a
genuine fix for the problem documented above.

Their shipped `leak_audit/` is weak evidence *on its own* — `hits.m8` is 0 bytes and all 884
rows of `per_test_chain.csv` are `fident=0, qcov=0, tcov=0, leaks=0`, which is
indistinguishable from a search that silently failed. It happens to be corroborated by my
independent check (nothing above 30.6% is expected to produce mmseqs hits at default
settings), and the surviving `tmp_l1uwiq1f/` query/target DBs are non-empty, so the search did
run. But an empty result file should never be the sole evidence of a clean split.

**Broken: fold balance.**

| dir | csv fold | rows | complexes |
|---|---|---|---|
| fold_1 | 0 | **4066 (61%)** | **211** |
| fold_2 | 1 | 922 | 13 |
| fold_3 | 2 | 387 | 8 |
| fold_4 | 3 | 245 | **1** (`3BT1`) |
| fold_5–10 | 4–9 | 168–170 each | 10–27 |

Consequences:
- fold_1 tests on 61% of the data while training on the remaining 39% (2,565 rows). Its score
  is not comparable to any other fold's.
- fold_4 tests on a **single complex** (`3BT1`, 245 mutations). A Spearman there is a
  within-complex ranking correlation, not cross-complex generalization — a different quantity
  from what the other folds measure.
- folds 5–10 each test ~2.5% of the data.
- **Do not report a mean across these 10 folds**, and do not pool predictions (the pool would
  be 61% fold_1, produced by a model trained on 39% of the data).

Cause: TM-score ≥ 0.5 is the "same fold" threshold, so under single linkage most globular
complexes chain into one giant component — 211 of 335 complexes (63%). The size profile (one
huge cluster, nine small, one of size 1) is what **leave-one-cluster-out over ~10 structural
clusters** looks like, not a balanced GroupKFold.

Options: report per-fold with N and never average; or hold the nine small clusters out
together against the mega-cluster as train; or break the mega-cluster with complete/average
linkage or a stricter TM cutoff.

Housekeeping: `tm50/leak_audit/tmp_*/` are committed mmseqs scratch dirs, several containing
only dangling `latest` symlinks. That shouldn't be in git.

### README bug

`README.md:50` passes `--folds_dir data/cross_validation_folds_final`, but
`evaluate_saved_models.py:214` expects `fold_N/` directly underneath. That path has only
`*_percent/` subdirs, so the documented command raises `FileNotFoundError`. It needs
`--folds_dir data/cross_validation_folds_final/60_percent` to reproduce the released models.

Analysis scripts for the above are in this session's scratchpad only (not committed).

## Prior findings (from earlier session notes, pre-purge)

- Linear probe on PLM embeddings: burial R²=0.40, SASA R²=0.46 (well predicted);
  interface R²=0.07–0.13 (poorly predicted)
- Key insight: PLMs lack interface information — ProtBFF's main value is injecting this
  missing prior

These numbers came from analyses in the now-deleted `analysis/`; the code that produced
them was local-only and is gone.

## Environment

- Python: `/n/home02/wangdz/.conda/envs/esm3/bin/python` (use the full path in SLURM scripts)
- Prefer the `shakhnovich` partition for CPU jobs; see the global `~/.claude/CLAUDE.md`
  for partition and memory-request guidance

---

# 2026-08-20 — Pulled ProtBFF-Private; MVA splits, PLOS re-run notebook, RDE/DDAffinity baselines

## Remotes / branches

`private` remote added → `https://github.com/Jfeldman34/ProtBFF-Private.git`. Local `main`
fast-forwarded 34e1a8e1 → **5e64841e** (clean FF, additions only).

| ref | head | date | contents |
|---|---|---|---|
| `origin/main` (public ProtBFF) | 8a4af15c | 08-13 | has the MVA splits, not the notebook |
| `private/main` | 5e64841e | 08-17 | + `plos_rerunning (1).ipynb` |
| `private/antoine/rde-ddaffinity-mva` | 8e5012bc | 08-18 | 3 commits off 8a4af15c; `benchmarks/` |

## New data: `data/cross_validation_folds_mva/{30,40,50,60,80,90}_percent/`

"mutated-vs-all" homology-aware splits (Jonathan, commit 8a4af15c). Each threshold has
`clusters.tsv` (335 complexes → cluster_id), `seq_edges.tsv` (explicit pairwise
similarity edges), `folds_XXpct.csv` (6631 rows + fold column), `fold_{1..10}/{train,test}_complex_ids.txt`.

Per-fold **test row** counts (sum 6631):

| thr | f1 | f2 | f3 | f4–f10 |
|---|---|---|---|---|
| 30% | 3410 | 456 | 387 | 336–363 |
| 40% | 2069 | 1149 | 456 | ~422 |
| 50% | 2051 | 1149 | 456 | 425 |
| **60%** | **1955** | **1149** | **456** | **438–439** |
| 80% | 1205 | 1149 | 652 | ~518 |
| 90% | 1149 | 610 | 609 | 609 |

Much better balanced than `tm50` (which was 4066/61% in fold 1) but fold 1 still dominates
below 80%. The 60% MVA fold sizes match exactly what both the notebook and Antoine's
benchmarks used → the committed splits ARE the ones behind all the numbers below.

**Not committed**: the generator. Notebook paths point at
`jonathanfeldman/test_protbff/Protbff_updates/clustering_fixes/` (`split_mva/`,
`shuffle_control.py`, `04_cluster_report.py`). Only outputs are in git — the leakage-fix
code itself is unreviewed and lives in one person's scratch dir. Get it committed.

**Name collision**: MVA `60_percent` ≠ `cross_validation_folds_final/60_percent`
(the old leaky CD-HIT one). Fold 2's test sets overlap in 0 entries.

## `plos_rerunning (1).ipynb` (26 cells, run interactively, left unfinished)

Paper model re-run under honest splits. Model = cross-attention over 5 score-scaled 768-d
blocks (3840-d), ddG head + ilDDT auxiliary head.

**Part A (cells 0–12), original protocol, random 10% val:**

| split | pooled Pearson | pooled Spearman | per-complex mean r |
|---|---|---|---|
| tm50 | 0.264 | 0.268 | 0.244 |
| tm70 | 0.281 | 0.289 | 0.238 |
| tm90 | 0.294 | 0.300 | 0.261 |
| MVA 60% | 0.294 | 0.314 | 0.261 |

Every honest split lands in 0.26–0.29 pooled. Insensitive to which one → the split choice
is not what matters; the leaky original was.
- MVA60 mean-of-folds r = **0.401**, pooled = **0.294**, fold-z-scored pooled = 0.347.
  That 0.11 gap = per-fold offset/scale disagreement between the 10 fold models.
- tm90 vs MVA60 predictions correlate r=0.78 despite only 33% of rows sharing a fold number.
- ilDDT auxiliary head is **dead**: pooled ilDDT Pearson 0.02–0.03 on every split.
- **Cell 9 (shuffle control) was interrupted** — folds were written, s0 got to fold 2, and
  the `=== leakage test (pooled, excl. fold 1) ===` comparison never printed. The
  size-matched random-split null for tm90 does not exist yet. This is the missing number.

**Part B (cells 13–24), tightened protocol** — cluster-aware val (whole clusters held out
of train), epoch chosen on val only (patience 15), sweep selected on VAL Pearson only:

- tuned `{ilddt_weight 0.25, lr 2e-4}`: mean VAL r 0.4516, **mean TEST r 0.3645** (pooled 0.3012)
- baseline (paper settings, same protocol): mean VAL r 0.4554, **mean TEST r 0.3861**
- → **tuning lost 0.02 on test**; the paper's hyperparameters were already fine.

**Ablation (cell 24), mean test r over folds:**

| variant | r |
|---|---|
| drop burial | 0.389 |
| full (5 scores) | 0.386 |
| drop SASA / drop lDDT | 0.369 |
| drop dihedral | 0.359 |
| **drop interface** | **0.352** ← largest single-score loss |
| only burial | 0.337 |
| only dihedral | 0.311 |
| only interface | 0.303 |
| all blocks → mean (no diversity) | 0.288 |
| only lDDT | 0.263 |
| only SASA | 0.195 |

Score *diversity* is worth ~0.10 (0.386 vs 0.288). Interface is the most load-bearing
single score — consistent with the old linear-probe finding that PLMs lack interface info.
Burial is redundant (dropping it is a wash/slightly better). Caveat: `SCORE_NAMES` in the
cell is marked "VERIFY vs merge_scores.py" — the ablation labels are **unverified**.

Notebook is unfinished: cells 23/25 empty, cell 24 says "paste after existing cells of
protbff_tuning.ipynb", no saved figures.

## Branch `antoine/rde-ddaffinity-mva` — RDE + DDAffinity baselines on MVA 60%

New tree `benchmarks/` (RESULTS.md, README.md, EMBEDDINGS.md, ddaffinity/BUGS.md,
analysis/compute_metrics.py, 61 per-checkpoint prediction CSVs, 22 MB).

Same 10 MVA60 folds, same 6631 entries:

| model | mean Pearson | mean Spearman | mean AUROC | pooled |
|---|---|---|---|---|
| RDE (30k iters) | 0.393 ± 0.087 | 0.354 | 0.651 | 0.375 |
| DDAffinity (ep 75) | 0.340 ± 0.142 | 0.293 | 0.631 | 0.325 |

- **Metric convention flips the ProtBFF ordering**: mean-of-folds ProtBFF 0.386–0.401 ≈ RDE
  0.393 > DDAffinity 0.340; pooled ProtBFF 0.294 < DDAffinity 0.325 < RDE 0.375. Decide the
  convention before quoting anything (not like-for-like — different checkpoint-selection rules).
- **Never quote sign-accuracy**: 76.9% of entries are destabilising, so constant
  "destabilising" = 0.756, beating RDE (0.705) and DDAffinity (0.684). Use AUROC.
- RDE hadn't converged at 30k (still improving) → 0.393 is a floor. RDE is wildtype-only
  (frozen upstream `RDE.pt` + trained head); DDAffinity also gets FoldX mutant structures,
  so part of the gap is inputs, not modelling.
- tm50 confirmed badly behaved from their side too: sd across folds 0.167 vs 0.087 on MVA.
- **DDAffinity needs 3 fixes to train at all** (`ddaffinity_fixes.patch`): (1) per-chain
  parser regression from `5ac2529` breaks all DDAffinity training from HEAD on any split;
  (2) `--resume` NameError; (3) no mid-training checkpoint. Also LMDB cache poisoning on
  interrupted builds, and requirements.txt numpy 1.24.3 vs biopython `np.bool`.
- Embeddings for ProtBFF: **out-of-fold** (each complex embedded by the fold model that held
  it out), (128,128) float32 patch around the mutation + resmap pkl, 6631/6631 verified,
  on Modal volume `rde-ppi-data` (`supercomputers-and-friends`), ~776 MB / ~1.1 GB tars.
  Stacking caveat: fold-N *train* features come from models that saw fold-N test complexes;
  `--splits` re-run at 10x cost for a bulletproof number. Test embeddings are clean either way.
  `extract_ddg_network_embeddings.py` hardcodes `threshold = 60` (~line 397).
- Weights not committed; RDE keeps only the newest checkpoint, so the iter-26k peak is gone.

## Open items

1. Finish the shuffle control (cell 9) — the leakage null is still unmeasured.
2. Commit `clustering_fixes/` (split generator + shuffle_control.py + 04_cluster_report.py).
3. Verify `SCORE_NAMES` order against `merge_scores.py` before using the ablation.
4. Pick pooled vs mean-of-folds; the two orderings vs RDE disagree.
5. `main` still tracks `origin/main` (public, 1 behind). Repoint to `private/main` if the
   private repo is now the source of truth.

## 2026-08-20 (same session) — symmetry probe: the antisymmetric readout is throwing away signal

`sym_test.py` / `sym_test2.py` (in repo root, untracked). Ridge regression on the *same*
cache (`skempi_score_cache.npz`), the *same* MVA 60% folds, no NN. Split the input pair into
`D = Xf - Xr` (the only thing the model's readout can express) and `S = (Xf+Xr)/2`:

| features | target | mean-of-folds r | pooled r |
|---|---|---|---|
| D (antisymmetric) | ddG | 0.355 | 0.199 |
| **S (symmetric)** | **ddG** | **0.418** | 0.317 |
| **[D \| S] (both)** | **ddG** | **0.444** | 0.314 |
| D (antisymmetric) | ilDDT | 0.307 | — |
| **S (symmetric)** | **ilDDT** | **0.531** | — |

(alpha=1000; stable over 100–10000: [D|S] 0.444/0.430. alpha NOT val-selected — redo properly.)

Compare: ProtBFF 0.386, RDE 0.393, DDAffinity 0.340 mean-of-folds on these same folds.
**A plain ridge on both parts (0.444) beats all three.**

Cause: `DDGPredictor.forward` returns `(head(mlp(xf)) - head(mlp(xr)))/2` — hard-wired
antisymmetric in the pair. It can only see `D`, which is the *weaker* half of the signal.

- Explains the dead ilDDT head: ilDDT ∈ [0,1], mean 0.975, sd 0.053 — an exchange-**symmetric**
  quality score being predicted by a strictly antisymmetric readout. Achievable r is ~0.53
  (symmetric ridge); the model gets 0.02. It is structurally incapable, not undertrained.
- Caveat: ddG *is* physically antisymmetric, so an S-using model predicts the same ddG for a
  mutation and its reverse. Part of S's advantage is likely the 76.9% destabilising class
  bias (perturbation magnitude → ddG). This is a scientific trade-off, not a free win.

**Zero-code-change way to test it**: the readout is antisymmetric in `(xf, xr)`, so feeding
`(Xf, -Xr)` makes it *symmetric* — same architecture, one line in the data loader.
Train one model on `(Xf, Xr)` and one on `(Xf, -Xr)` and average → approximates `[D|S]`.

## Broken sweep (do not trust "tuning doesn't help")

- Cell 20: `sweep(..., folds_subset=(0,0))` → `folds[0]` **twice**, i.e. selection on fold 1
  alone, n=1 duplicated.
- Cell 19's committed `GRID` is a single point `{ilddt_weight 0.3, lr 5e-5}`, but cell 21's
  "winning overrides" is `{ilddt_weight 0.25, lr 2e-4}` — a different sweep.
- Execution counts: cell 21 = 38, cell 20 = 44 → **cell 21 ran before cell 20**. The tuned
  result does not come from the committed grid.

So mean TEST 0.3645 (tuned) vs 0.3861 (baseline) is not evidence that the paper's
hyperparameters are optimal. A real multi-fold sweep has not been run.

## 2026-08-20 — ESM3 (structure-conditioned) swap-in for ProSST

### Is the code here? No — the pipeline is ProSST, not ESM

`data_pipeline/embedding_pdb_full.py` loads **`AI4Protein/ProSST-2048`** (768-d), with
structure tokens from `tokenize_pdb_full.py` + the GVP quantizer in `data_pipeline/prosst/`.
ProSST is already structure-conditioned, so "PLM + structure" is the current design; ESM3 is
a *different* structure-conditioned PLM, not a new capability.

The only ESM artifacts that existed are **ESM2-650M** (1280-d → 5×1280 = 6400) caches for the
antibody benchmarks: `model_benchmarking/score_caches/{9lyp,ace2}_esm_score_cache.npz`.
No SKEMPI ESM cache, and the ESM2 embedding script is not in this repo (it lived in
`jonathanfeldman/ProSST_PPI-main/`).

### Everything else needed IS here

- `data/wildtype/` 6993 PDBs, `data/optimized/` 6993 FoldX-relaxed mutant PDBs, `data/lddt_dir/`
- `data/SKEMPI2_filtered_final.csv` (6631 rows)
- Jonathan's tree has the reusable intermediates:
  `…/jonathanfeldman/test_protbff/ProtBFF/data/merged_output/merged_*.npz` (**6632 files**,
  per-residue scores + ProSST Xf/Xr) and `data/{wildtype,optimized}_embeddings_2048/`
- ESM3 weights already cached: `~/.cache/huggingface/hub/models--EvolutionaryScale--esm3-sm-open-v1`

**Scores are embedding-independent**, so calculate_all_scores.py does NOT need re-running —
only the embeddings change.

### New code (untracked, in repo root / data_pipeline)

| file | what |
|---|---|
| `data_pipeline/esm3_embed_pdb.py` | per-residue ESM3 embeddings, (L, 1536) `.npy` |
| `data_pipeline/esm3_embed.sbatch` | `sbatch [--array=…] esm3_embed.sbatch <pdb-dir> <out-dir>` |
| `data_pipeline/rebuild_merged_esm3.py` | swaps Xf/Xr in merged NPZs, keeps every score array |

**Row alignment is the whole game.** `merge_scores.py` silently skips any structure whose
embedding length != score length, so the ESM3 rows must follow the exact BioPython order in
`embedding_pdb_full.parse_pdb_sequence` (model→chain→residue, `residue.id[0]==' '`, standard
AA only, after `reassign_empty_chain_atoms`). `esm3_embed_pdb.py` imports those two helpers
directly rather than re-deriving the order.

Do NOT use `ProteinComplex.from_pdb(...).as_chain()`: it asserts single-chain and dies on every
complex. And `ProteinComplex`'s own residue order disagrees with BioPython's — checked 25
random wildtype PDBs, 24 matched, `658_4CPA` (3 chains) differed from position 307 at equal
length, i.e. a chain-order difference. Hence building atom37 by hand.

Multi-chain handling: chains joined with a `'|'` chainbreak (NaN coord row), so ESM3 sees the
interface; the break rows are dropped after `model.logits(..., return_embeddings=True)` and
BOS/EOS stripping.

### Jobs (gpu_test)

- **40785631** validation, 20 wildtype structures — **PASSED**: 20/20 written, 0 failed,
  all `(L, 1536)`, all finite, every L matching its `merged_*.npz` score length. Cancelled after.
- **40787326** `esm3_wt` → `data/wildtype_embeddings_esm3/` (6993)
- **40787328** `esm3_opt` → `data/optimized_embeddings_esm3/` (6993)

11 h limit, 24G, 1 MIG GPU each. Re-runnable: existing outputs are skipped unless `--overwrite`.
Note ESM3-1.4B needs ~6 GB just to load — a CPU test on the login node gets OOM-killed.

### Remaining steps once embeddings land

```bash
python data_pipeline/rebuild_merged_esm3.py \
  --merged_dir /n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data/merged_output \
  --wt_embedding_dir data/wildtype_embeddings_esm3 \
  --opt_embedding_dir data/optimized_embeddings_esm3 \
  --output_dir data/merged_output_esm3
python data_pipeline/merge_scores.py --merged_dir data/merged_output_esm3 \
  --output_cache model_benchmarking/score_caches/skempi_esm3_score_cache.npz
```
Then `run_cv_v2` from the notebook on `data/cross_validation_folds_mva/60_percent` with
**`embed_dim=1536`** (cache becomes (6631, 7680)). Compare against ProSST 0.386.

## CORRECTION: the notebook's ablation labels are wrong

`merge_scores.py:228` and `data_pipeline/README.md` both give the block order as
**interface, burial, lDDT, SASA, dihedral** (and lDDT is inverted, `1 - lddt`, before pooling).
The notebook's `SCORE_NAMES = ['interface','burial','dihedral','SASA','lDDT']` swaps positions
2 and 4 — its own `# VERIFY vs merge_scores.py` comment was justified. **Exchange the dihedral
and lDDT rows** of the ablation table:

| corrected label | mean test r |
|---|---|
| drop burial | 0.389 |
| full (5 scores) | 0.386 |
| drop SASA | 0.369 |
| **drop dihedral** | 0.369 |
| **drop lDDT** | 0.359 |
| drop interface | 0.352 |
| only burial | 0.337 |
| **only lDDT** | 0.311 |
| only interface | 0.303 |
| all → mean | 0.288 |
| **only dihedral** | 0.263 |
| only SASA | 0.195 |

Single-score ranking is burial > lDDT > interface > dihedral > SASA (not burial > dihedral >
interface > lDDT > SASA as the notebook printed).

## 2026-08-20 — Tier-1 + Tier-2 tuning run on the fixed MVA 60% split

**"Redo train/test split"** = run on the homology-aware **MVA 60%** split (the leakage fix),
with a proper 3-way split: MVA train/test + **cluster-aware VAL** held out of train
(whole clusters, using `data/cross_validation_folds_mva/60_percent/clusters.tsv`, 111 clusters).

Code (untracked): `tuning_v1/protbff_train.py` (self-contained; paper architecture UNCHANGED),
`tuning_v1/run_sweep_final.sbatch`. Uses ProSST cache `skempi_score_cache.npz` (3840-d).

**Tier-1 protocol**: epoch selected on VAL Pearson; refit on train+val at selected epoch;
5-seed ensemble (avg predictions); per-fold affine calibration fit on VAL → applied to test
(affects POOLED metric only — per-fold Pearson/Spearman are affine-invariant).
Target standardized per fold (stabilizes optimization).

**Tier-2 sweep** (random search, 24 configs + AdamW baseline, selected on mean VAL Pearson
over 3 representative folds [1,2,5]): AdamW weight_decay {1e-4,1e-3,1e-2}, num_hidden {1,2,3},
ilddt_weight {0,0.2}, loss {mse,huber}, lr {5e-5,1e-4,2e-4}, batch {32,64}, dropout_scale {0.7,1.0}.

**KEY GOTCHA found**: standardizing the target makes the ilDDT auxiliary loss a *real* 0.2
weight. In the notebook it was effectively ~0 (ilDDT sd 0.053 vs ddG sd 2.07, so
0.2·MSE(ilddt) ≈ 0.0001·MSE(ddg)). So the notebook's "ilddt_weight=0.2" was a no-op; here it
genuinely pollutes the shared pooling → sweep should pick ilddt_weight=0. This is self-correcting.

**Early sweep trend (as expected, regularization-first)**: best configs have wd=0.01 + num_hidden=2
+ huber. AdamW baseline (paper HPs) valP=0.385 on the 3 hard folds; wd=0.01/nh=2/huber → valP=0.409.

**Jobs**: 40796197 (gpu_test, sweep→select→final). Metrics reported: mean-of-folds & pooled,
Pearson AND Spearman, baseline vs tuned. Output `tuning_v1/out/{sweep,best_config,final}.json`.

**ESM3 embeddings DONE**: 40787326/40787328 COMPLETED (26 min each). wildtype + optimized ESM3
embeddings written to `data/{wildtype,optimized}_embeddings_esm3/` (~6911/6993 each; ~82
structures failed parsing — investigate before ESM3 model run, but merge skips missing gracefully).

## 2026-08-20/21 — Tier-1+Tier-2 tuning RESULTS (job 40796197, COMPLETED 1h04m)

Sweep winner (val-selected, 3 folds): lr=2e-4, **wd=0.01**, bs=64, **nh=2**, ilw=0.2, **huber**, ds=0.7.
All top-5 configs had wd∈{1e-3,1e-2} + nh=2 + ds=0.7 → **regularization-first hypothesis confirmed**
(paper used wd=1e-5, nh=3). This matches the ridge finding that the model is under-regularized.

Final 10-fold (MVA 60%), Spearman + Pearson:

| condition | mean ρ | mean r | pooled ρ | pooled r |
|---|---|---|---|---|
| baseline (paper HPs, AdamW) | 0.315 | 0.335 | 0.267 | 0.245 |
| tuned + refit + 5-seed ens + calib | 0.310 | 0.348 | **0.289** | 0.267 |

→ pooled improved (+0.022 ρ, +0.022 r) but **mean-of-folds flat** (ρ −0.005). refit+ensemble
mostly bought fold-to-fold *consistency* (pooled), not per-fold accuracy.

### TWO ISSUES in this run (do not quote these as final)
1. **Baseline depressed vs notebook.** Notebook Part-B paper-HP baseline = 0.386 mean r; mine = 0.335.
   Cause: I **standardize the target per fold**, which turns the ilDDT aux loss (ilw=0.2) from an
   effective no-op (ilDDT sd 0.053 vs ddG 2.07) into a real 0.2 weight on an *unlearnable* task
   (symmetric target through antisymmetric readout) that pollutes the shared pooling. So tuning
   happened inside a depressed regime; 0.348 does NOT beat 0.386.
2. **Calibration is buggy under refit** — `pooled+calib` came out WORSE (0.248). With refit on, the
   affine map is fit on the early-stop model's val but applied to the *refit* model's test preds
   (different model → scale mismatch). Fixed in diag mode by not refitting (val & test same model).

### Clean diagnostic (job 40811975, gpu_test) — user chose "clean re-run"
Added `standardize` toggle + `diag` mode to `tuning_v1/protbff_train.py`. 5 conditions, 10 folds,
1 seed, valid calibration:
- A paper+std+ilw0.2 (reproduce my 0.335)   - B paper+NOstd+ilw0 (~notebook, expect ~0.386)
- C paper+NOstd+ilw0.2 (isolate aux)        - D tuned+NOstd+ilw0 (reg gains on recovered baseline)
- E tuned+NOstd+ilw0.2
Output `tuning_v1/out/diag.json`. RESULTS PENDING.

### DIAGNOSTIC RESULT (job 40811975) — tuning did NOT help; paper HPs win

| condition | mean ρ | mean r | pooled ρ | pooled r |
|---|---|---|---|---|
| A paper+std+ilw0.2 (my depressed baseline) | 0.315 | 0.335 | 0.267 | 0.245 |
| **B paper+NOstd+ilw0 (clean, ≈notebook)** | **0.360** | **0.389** | **0.306** | 0.269 |
| C paper+NOstd+ilw0.2 | 0.350 | 0.377 | 0.296 | 0.263 |
| D tuned+NOstd+ilw0 | 0.334 | 0.377 | 0.265 | 0.236 |
| E tuned+NOstd+ilw0.2 | 0.340 | 0.383 | 0.275 | 0.250 |

Conclusions:
1. **Standardization was the whole confound.** A→B: +0.054 mean r, +0.045 mean ρ just by turning
   off target standardization + ilDDT aux. B (0.389) reproduces the notebook's 0.386.
2. **Tier-2 tuning gave a FALSE signal.** The sweep ran only in the standardized regime, where
   heavy regularization *compensated* for the depression. In the clean regime the "tuned" config
   (wd=0.01,nh=2,huber) is WORSE than paper HPs on every metric: D/E mean r 0.377–0.383 vs B 0.389,
   mean ρ 0.334–0.340 vs B 0.360. **Paper hyperparameters are already optimal among all tested.**
3. **ilDDT aux is mildly harmful even without std** (B 0.389 > C 0.377), consistent with the
   dead-head argument — but the effect is small (−0.012), not the −0.054 the sweep saw.
4. Tier-1 refit+5seed+calib (earlier run, standardized) got pooled ρ 0.289 — still below clean
   single-seed B (0.306). refit/ensemble did not beat the clean paper baseline; calibration bugged.

**HONEST HEADLINE — ProtBFF on fixed MVA 60%, paper HPs + honest protocol (cluster-aware val,
epoch on val Pearson):  mean-of-folds Spearman 0.360 / Pearson 0.389;  pooled Spearman 0.306 /
Pearson 0.269.**  ≈ RDE (0.393 r), > DDAffinity (0.340 r). HP tuning does not improve this; the
real lever is the symmetric-readout architecture change (ridge [D|S] hit 0.444), not hyperparameters.

Config B is the recommended setup: AdamW ok, lr 1e-4, wd 1e-5, nh 3, MSE, dropout as paper,
**standardize=False, ilddt_weight=0**, epoch selected on val Pearson.

## 2026-08-21 — Architecture upgrade: symmetric-aware readout (job 41512694, gpu_test)

Rationale (established): antisym readout (head(h_f)-head(h_r))/2 sees only D=x_f-x_r; ridge showed
S=(x_f+x_r)/2 is the STRONGER half (S 0.418 > D 0.355 alone), and [D|S] ridge=0.444 vs net 0.389.
NN beats ridge on D-only (0.389>0.355) → the net works; the readout blindfolds it to S.

Code `tuning_v1/protbff_arch.py`: same pooling + shared per-branch MLP, configurable readout over
h_f=mlp(pool(x_f)), h_r=mlp(pool(x_r)):
  antisym (=baseline) | dual = head_a(h_f-h_r)+head_s(h_f+h_r) | concat_ds = head([h_f-h_r ; h_f+h_r])
  | concat_pair = head([h_f ; h_r])
Clean protocol (condition B): NO target standardization, ilddt_weight=0, cluster-aware val, epoch on
val Pearson. Rigor: 5 seeds/fold (report seed-ensemble per-fold + single-seed stability), mean±SEM
over 10 folds, pooled, Pearson AND Spearman, PAIRED Wilcoxon vs antisym baseline, plus a ridge[D|S]
reference with alpha val-selected per fold (fair, unlike the earlier fixed-alpha probe).
Smoke-tested OK on CPU. Output `tuning_v1/out/arch.json`.
Physical caveat to carry: non-antisym readouts predict same ddG for a mutation & its reverse; part of
S's gain is the 77% destabilising class bias. Gain is real for SKEMPI corr but trades physical consistency.

### RESULTS (job 41512694 COMPLETED 2026-08-24; log `slurm_logs/arch_41512694.out`)

Symmetric-aware readouts beat the paper's antisymmetric one on EVERY metric. `dual` is best.
5 seeds/fold, 10 folds, MVA 60%, clean protocol (NO std, ilw=0, cluster-aware val), seed-ensembled:

| readout | mean-of-folds P | mean-of-folds S | pooled P | pooled S |
|---|---|---|---|---|
| antisym (paper baseline) | 0.422 ± 0.035 | 0.375 | 0.302 | 0.324 |
| **dual = head_a(h_f−h_r)+head_s(h_f+h_r)** | **0.451 ± 0.037** | **0.412** | **0.329** | **0.368** |
| concat_ds = head([h_f−h_r ; h_f+h_r]) | 0.444 | 0.405 | 0.315 | 0.347 |
| concat_pair = head([h_f ; h_r]) | 0.442 | 0.404 | 0.326 | 0.361 |
| ridge[D\|S] reference (alpha val-selected) | 0.434 | 0.410 | 0.303 | 0.352 |

1. **Hypothesis confirmed.** dual vs antisym: +0.029 mean P, +0.037 mean S, +0.027/+0.044 pooled.
   The antisym readout was discarding the stronger (symmetric) half of the signal, as `sym_test.py` showed.
2. **NN now beats linear.** dual (0.451) > ridge[D|S] (0.434) on mean P — once the net can see S it
   exploits it better than ridge. Previously antisym net (0.422) trailed ridge.
3. **NOT significant at n=10.** Paired Wilcoxon vs antisym: dual p=0.193 (Pearson), **p=0.084 (Spearman)**;
   concat_ds/concat_pair weaker. Direction consistent across all metrics but fold sd ≈0.06 → underpowered.
   Real & consistent gain, not yet proven. To firm it up: more seeds won't help (fold variance dominates);
   need more folds or a paired bootstrap, or accept it as a consistent-direction improvement.
4. Physical caveat stands (same-ddG-for-reverse; 77% destabilising class bias inflates S).

**PROJECT HEADLINE:** HP tuning is a dead end (paper HPs optimal, job 40811975). The architecture change
is the ONLY lever that moved the number: dual readout **0.451 mean P** vs paper antisym 0.389/0.422,
now ≥ RDE (0.393) and > DDAffinity (0.340) on the honest MVA-60 split. Recommend `dual` readout for the
paper, reported with the physical-antisymmetry caveat + the n=10 significance qualifier.

## 2026-08-30 — Tier-1 improvements: ESM3 swap-in run + RDE ensemble (jobs 43086164/166/175)

Two ready-to-run levers from the improvement menu. New untracked code in `tuning_v1/`:
`build_esm3_cache.sbatch`, `oof_preds.py`+`run_oof.sbatch`, `run_esm3_arch.sbatch`,
`ensemble_rde.py`; plus additive `--embed_dim` arg on `protbff_arch.py` (default 768).

### Alignment facts established this session (reusable)
- **ProSST cache `skempi_score_cache.npz` is in EXACT SKEMPI/master row order**: `y` and complex id
  match `folds_60pct.csv` element-wise, 0/6631 mismatches. So cache row i == master row i == folds CSV row i.
- **RDE `results_*.csv` is in a DIFFERENT order** (6540/6631 positional complex mismatches) AND
  `(complex,mutstr)` has 886 dups → NOT a join key. Join RDE by `(complex, round(ddG,3))` with
  group-mean of `ddG_pred` (ddG is shared SKEMPI ground truth → exact join, residual ~0).
- **master `fold` column (0-indexed) == fold_k dirs** (fold_dir k = mastercol+1), 0/6631 mismatch.
  Fold sizes: 1955,1149,456,439×5,438×2.
- `merge_scores.py` now saves `mutations=None` (older ProSST cache had `<U4` mutations — don't rely on it).

### Job A (43086164, shakhnovich CPU, 8G): build ESM3 cache
`rebuild_merged_esm3.py` (source merged_output from jonathanfeldman, 6631 files; ESM3 embeds (L,1536))
→ `data/merged_output_esm3/` → `merge_scores.py` → `skempi_esm3_score_cache.npz` (expect (~6567, 7680)).
Preflight: protein_id `0_1CSE` matches embed filenames; **6567/6631 rows survive** (64 missing embeds, ~1%).
NOTE for clean compare: ESM3 runs on 6567 rows vs ProSST 6631 — if result is close, re-run ProSST on the
matched 6567 subset.

### Job B (43086166, gpu_test, 16G): ProtBFF dual OOF preds on ProSST
`oof_preds.py` reruns clean-protocol dual (5 seeds,10 folds) and dumps per-entry OOF test preds →
`tuning_v1/out/oof_dual_prosst.csv` (idx,fold,y,pred). Doubles as a reconfirm of dual=0.451.

### Job C (43086175, gpu_test, 24G, afterok:A): ESM3 arch run
`protbff_arch.py --embed_dim 1536 --variants antisym,dual` on ESM3 cache → `tuning_v1/out/arch_esm3.json`.
Compare ESM3-dual vs ProSST-dual (0.451) and ESM3-antisym vs ProSST-antisym (0.422).

### Downstream (CPU, after B): `ensemble_rde.py` — DONE ✅
Joins OOF dual + RDE 30k, equal-weight per-fold z-avg ensemble (rank-avg as robustness).
Job B (oof_dual) COMPLETED, reproduced dual exactly (meanP 0.4510/meanS 0.4115/pooled 0.3286).

**RESULT (MVA-60, n=6631, all metrics):**
| model | meanP | meanS | mAUROC | poolP | poolS |
|---|---|---|---|---|---|
| RDE (30k) | 0.393 | 0.354 | 0.651 | 0.374 | 0.358 |
| ProtBFF dual | 0.451 | 0.412 | **0.668** | 0.329 | 0.368 |
| **ensemble z-avg** | **0.475** | **0.435** | **0.683** | **0.415** | **0.412** |
| ensemble rank-avg | 0.436 | 0.433 | 0.682 | 0.389 | 0.408 |

- **z-avg ensemble beats both members on ALL five metrics** (+0.024 meanP over best member;
  fixes ProtBFF's pooled weakness 0.329→0.415 via RDE's cross-fold calibration).
- **Missing ProtBFF AUROC filled: 0.668** — best single model (RDE 0.651, DDAffinity 0.631).
- RDE reproduces Antoine exactly (0.393/0.651/0.374) → join + folds verified correct.

**JOIN-BUG LESSON (fixed):** first pass joined RDE by `(complex, round(ddG,3))` WITHOUT the
mutation string → different mutations in a complex with coincidentally-equal rounded ddG collided
and got group-averaged, inflating RDE to 0.411/pooled 0.394. Fix: key on
`(complex, mutstr/Mutation(s)_cleaned, round(ddG,3))`. Always sanity-check a recomputed baseline
against its known value (RDE 0.393) before trusting a derived number. Output `tuning_v1/out/ensemble_rde.json`.

All 3 jobs verified queued via squeue at submit (A+B running, C pending-dependency).

## 2026-08-30 — Overleaf access (git bridge) + manuscript cross-check

### Overleaf access (works, reusable)
- Project id `6a94b1bb408dd5457828a8ee`; git remote `https://git.overleaf.com/<id>` (git bridge).
- **Git auth token** stored at `~/.config/overleaf_token` (chmod 600), askpass helper
  `~/.config/overleaf_askpass.sh` (chmod 700). Token is NOT in any `.git/config` or repo.
- Clone lives OUTSIDE ProtBFF at `/n/netscratch/shakhnovich_lab/Lab/jwang/protbff_overleaf`
  (sibling dir; `core.askpass` set locally so pull/push just work). Note: netscratch purges —
  re-clone if gone; token file in home dir survives.
- Manuscript = `plos_latex_template.tex` (623 lines, PLOS format). Figures + `Shakhnovich.bib` alongside.
- **Treat every push as an outward action — show diff, get approval first.** Nothing edited yet.

### Manuscript cross-check vs our measured results (READ before editing the paper)

Authors: Feldman, Maechler, Wang, Shakhnovich. Title: "Biophysically Grounded Deep Learning
Improves Protein–Protein ΔΔG Prediction."

**🔴 CRITICAL — Table 1 numbers look like the LEAKY split, not MVA.** Paper Table 1 reports
ProSST+ProtBFF **0.515/0.471**, RDE-Network 0.480, DDAffinity 0.485, ProMIM 0.486 (Pearson/Spearman).
Every honest MVA-60 number we measured is ~0.1 LOWER: ProtBFF 0.389 (paper arch)→0.451 (dual),
RDE **0.393**, DDAffinity **0.340** (mean-of-folds P). `0.515` appears nowhere in our runs; the whole
table is shifted +~0.1 = signature of a leakier split. Methods (§Model Training, §Dataset) say the
benchmark used "the 60% **CD-HIT** split" — but we proved that exact CD-HIT-on-concatenated-chains 60%
split is still leaky (14.6% test rows share a byte-identical chain w/ train); the fixed split we use is
**MVA/mmseqs**, which is where 0.451/0.393/0.340 come from. So the headline table appears to sit on the
very split the paper's own "Limitations of SKEMPI2" section argues against. **MUST resolve before
submission:** confirm which split produced 0.515 (ask Jonathan/Antoine), then either re-run Table 1 on
MVA (numbers drop ~0.1, story survives: ProtBFF ≈ RDE, > DDAffinity) or correct the Methods.

**🟠 Ablation discrepancies (same root cause — split).**
- Burial: paper says dropping it costs 0.044 (its #2 feature); our MVA ablation says dropping burial is
  a wash/slightly BETTER (0.389 vs full 0.386) — burial is redundant on the honest split. Only "interface #1"
  agrees. Paper drop-rank interface>burial>dihedral>SASA>lDDT; ours interface>lDDT>dihedral≈SASA>burial.
- ilDDT aux: paper says removing it hurts; our MVA says it's mildly harmful (0.389 w/o vs 0.377 w/) and
  it's a structurally dead head under the antisym readout.

**🟠 Antisymmetric readout stated as a virtue** (§Architecture eq. `(f(X_f)−f(X_r))/2` "to preserve
anti-symmetry"). Matches the code, so not an error — but it's exactly the design our dual readout
improves on (0.451 > 0.422). If the dual result goes in the paper, this section changes.

**🟡 Minor / rigor.**
- Table 1 labels swapped: "Pearson (ρ)" / "Spearman (r)" — conventionally r=Pearson, ρ=Spearman. Reviewer bait.
- No AUROC, no error bars/significance, metric convention (pooled vs mean-of-folds) unstated — our notes
  show convention flips ProtBFF-vs-RDE ordering; never quote sign-accuracy (77% destabilising baseline).
  The running ensemble job will give the missing ProtBFF AUROC.
- Cluster-count table (60%→136 clusters) is CD-HIT; MVA `clusters.tsv` has ~111 at 60% — different methods,
  reconcile.
- Consistent & fine: 335 complexes/6631 muts, architecture section matches code, loss `1.0·MSE(ddG)+0.2·MSE(ilddt)`.

## 2026-08-30 — "Run remaining paper models on MVA-60" (feasibility + launches)

Goal: every Table-1 model on the honest MVA-60 split (all EXCEPT the ESM2 size sweep).
User steer: use **Modal** ($1030 credits, workspace dianzhuo-wang) for the heavy retrains, but
verify the run first ("sure about the running" → smoke test, THEN Modal).

### Feasibility matrix (what exists, what's needed)
| model | on MVA? | path | effort |
|---|---|---|---|
| ProtBFF ProSST | ✅ done | dual 0.451 / antisym 0.389 | — |
| RDE / DDAffinity | ✅ done (Antoine) | 0.393 / 0.340 | — |
| ESM3+ProtBFF | ⏳ job C | embed_dim 1536 | — |
| **ESM2+ProtBFF** | 🚀 LAUNCHED | per-residue ESM2 `.pt` (L,1280) exist, 6631/6631 | LOW |
| ESM2 baseline (bare) | pending | `jonathanfeldman/esm2_embeddings.npz` (6977, 1280) pooled — plain readout on MVA | LOW |
| **ProMIM** | ❌ needs full MVA RETRAIN | cached embeds are FOLD-TIED (cmp fold_1≠fold_2) → can't reuse; only cluster sbatch, no Modal yet | HIGH |
| bare ProSST (scores=1) | ❌ | rebuild cache w/ scores=1 → arch | LOW-MED |
| RDE-Linear | ❌ | ridge on RDE OOF embeds (Modal vol `rde-ppi-data`) — pull + fit | MED |
| FoldX | ❌ | NO prediction outputs exist → needs FoldX compute pass (`compute_foldx_ddg.py`) | MED |

### ESM2 pipeline LAUNCHED (mirrors ESM3, verified)
- ESM2 per-residue embeds: `jonathanfeldman/esm2_embeddings/{wildtype,mutant}/<pid>.pt` (L,1280),
  6993 each, **6631/6631 coverage** (preflight, 0 missing). `<pid>.pt` matches merged protein_id.
- New: `data_pipeline/rebuild_merged_esm2.py` (.pt/1280 loader, else identical to esm3 rebuild).
- **Job 43089638** (shakhnovich CPU): rebuild → `data/merged_output_esm2/` → merge_scores →
  `skempi_esm2_score_cache.npz` (expect (6631, 6400)).
- **Job 43089639** (gpu, afterok:43089638): `protbff_arch.py --embed_dim 1280 --variants antisym,dual`
  → `tuning_v1/out/arch_esm2.json`. Compare vs paper ESM2 0.194→0.451 (leaky split).

### Baselines (bare encoder, no ProtBFF) — `tuning_v1/baseline_mva.py`
Reads a merged_output dir, max-pools the per-residue embedding diff (no scores), ridge [D|S] on MVA
folds (alpha val-selected). **ProSST baseline DONE: mean P=0.2757, S=0.2467, AUROC=0.6053, pooled P=0.150.**
vs ProSST+ProtBFF dual 0.451 → **ProtBFF gain on honest split = +0.175 mean P** (bigger than the paper's
leaky-split +0.087; priors help more when you can't memorize homologs — strengthens the thesis).
ESM2/ESM3 baselines: run the same script on merged_output_esm2/esm3 once those merges finish.

### ProMIM — CONFIRMED tractable on MVA (was the feared blocker)
- Reads folds via `train_complex_ids.txt`/`test_complex_ids.txt` (src/datasets/skempi.py:132) = MVA's
  exact format. Uses FROZEN pretrained encoders (`trained_models/PIM_BIM.pt`, `SIM.pt`) + processes
  structures per-fold from `pdb_dir`/`cache_dir`. The fold-tied `promim_embeddings/` (6.5G) is NOT
  consumed by training — red herring. So MVA = just swap fold_dir.
- ProMIM dir is READ-ONLY + 23G (mostly `logs_skempi/`). Selective copy (src+configs+trained_models
  476M+data 91M, skip logs/embeddings) → `/n/netscratch/.../jwang/promim_mva` (~570M, copy in progress).
- Config `configs/train/promim_ddg_skempi.yml`: fold_dir→OLD 40%, max_iters 5000, adam lr 3e-4, patch 128.
- NEXT: point fold_dir → `data/cross_validation_folds_mva/60_percent`, extract PDBs if needed
  (data/SKEMPI_v2/PDBs has only 3 + SKEMPI2_PDBs.tgz), env `jonathanfeldman/envs/promim`, smoke-test 1
  fold on cluster GPU, then 10-fold train. Modal optional (data is cluster-local, so cluster gpu is natural).

### ProMIM smoke FAILED — env is a netscratch-purge casualty (BLOCKER)
Job 43097975: `Fatal Python error: init_fs_encoding ... no codec search functions ... can't find
encoding`. Diagnosed: NOT env pollution (PYTHONPATH/HOME empty; fails under `env -i` too). The
promim env `jonathanfeldman/envs/promim` is **partially purged** — `lib/python3.8/encodings/` MISSING
(37 lib entries vs normal ~200; site-packages only 56 for a 200+ pkg requirements.txt). Interpreter dead,
site-packages gutted → NOT repairable, needs full rebuild.
- Rebuild is heavy: py3.8 + torch2.0/CUDA + torch_geometric + dgl 1.1.3 + atom3d + **custom `unicore`
  0.0.1** (Uni-Mol, needs building). `ProMIM/requirements.txt` is a full freeze. `promim.zip` (22GB) exists
  but listing was unusable.
- **Recommended path: rebuild on Modal** (clean container image from requirements, upload the 567M
  `promim_mva/` code+data+checkpoints to a volume, run 10-fold). Sidesteps the purged cluster env.
  All MVA wiring already done in `promim_mva/` (fold_dir→MVA, smoke config, PDBs extracted). DECISION PENDING.

### Still TODO
1. **ProMIM** — BLOCKED on env rebuild (see above). Modal container is the path; needs user go-ahead (real effort).
2. ESM2/ESM3 baselines (after merges); ESM2/ESM3+ProtBFF — ESM2 cache DONE, arch about to run; ESM3 cache finishing.
3. RDE-Linear (Modal embeds), FoldX (compute pass) — weak baselines, external deps, lowest priority.

## 2026-08-30 — DRAFT-UPDATE STRATEGY (honest split) — agreed plan, minimal-modification

Goal: update the Overleaf draft (`plos_latex_template.tex`) to the honest MVA-60 split WITHOUT rewriting.
Core framing: it's a **numbers swap, not a rewrite** — the paper's own "Limitations of SKEMPI2" section
already argues homology inflates numbers, so lower honest numbers CONFIRM the thesis and RESOLVE the current
internal contradiction (arguing CD-HIT-60 is leaky while reporting the headline table on it).

**Scope — CHANGES (small):** Table 1 numbers (one-for-one), Methods split paragraph (CD-HIT→MVA/mmseqs),
~3 number-bearing sentences (0.428→0.515 ProSST line; ablation ranking; ilDDT-helps line), ablation table.
**STAYS untouched:** intro, biophysical-score definitions, architecture section, discussion, SARS-CoV-2 section.

**THE KEY DECISION (two distinct updates — don't conflate):**
- (a) Minimal split-update = report the PAPER'S OWN model (antisymmetric readout) on MVA → ProSST+ProtBFF ≈ **0.39**
  (NOT 0.451). Architecture section unchanged.
- (b) Dual readout (0.451) is a NEW method contribution, changes the architecture section — NOT a split update.
- Consequence: on honest split the antisym model (~0.39) only TIES RDE (0.393) → "surpass SOTA" weakens to
  "matches". The **dual readout (0.451) RESTORES "surpass"** (>RDE). Fork: (a) minimal + soften "surpass"→"matches";
  or (a+b) keep "surpass" but add dual readout as a scoped addition. RECOMMEND (a) now, decide (b) once ESM dual
  numbers land. **User must also pick metric convention (mean-of-folds vs pooled) — flips ProtBFF-vs-RDE ordering;
  recommend mean-of-folds (what all models have, favors ProtBFF).**

**Mechanism (enforces minimal-modification):** wait for ALL honest numbers before editing Table 1 (one pass, not
piecemeal); wrap every changed value/sentence in the draft's existing `\revised{}` macro (review mode on) so
co-authors see the delta; ONE commit; SHOW USER THE DIFF BEFORE PUSHING (every Overleaf push = outward action).
Do NOT touch SARS-CoV-2 section or regen figures in this pass (Fig 2 threshold-sweep is already honest-split).
Hold ensemble/AUROC as SI/footnote at most (additions = over-modification risk).

### Live jobs (all verified in squeue at submit)
43086164 esm3_cache(R) · 43086166 oof_dual(R, slow ~26min) · 43086175 esm3_arch(PD,dep) ·
43089638 esm2_cache(PD) · 43089639 esm2_arch(PD,dep). Ensemble waiter still armed on oof_dual.
