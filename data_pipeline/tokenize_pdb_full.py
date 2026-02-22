from pathlib import Path
from prosst.structure.quantizer import PdbQuantizer
from tqdm import tqdm
import argparse
import os


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Quantize PDB structures with ProSST.")
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=repo_root / "data" / "optimized",
        help="Directory containing input PDB files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "optimized_tokens_2048",
        help="Directory to write tokenized FASTA files.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2048,
        help="Structure vocabulary size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing outputs.",
    )
    return parser.parse_args()


def quantize_pdb(
    pdb_file: Path, output_dir: Path, processor: PdbQuantizer, overwrite: bool
) -> None:
    pdb_id = pdb_file.stem
    output_path = output_dir / f"{pdb_id}_full.fasta"

    if output_path.exists() and not overwrite:
        print(f"Skipping {pdb_file.name}")
        return

    print(f"Processing {pdb_file.name}")
    result = processor(str(pdb_file), return_residue_seq=False)
    vocab_key = str(processor.structure_vocab_size)

    try:
        chain_seq = result[vocab_key][pdb_file.name]["struct"]
    except KeyError as exc:
        raise RuntimeError(
            f"Unexpected quantizer output for {pdb_file.name}"
        ) from exc

    structure_seq_str = ",".join(map(str, chain_seq))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write(f">{pdb_id}_full\n")
        handle.write(structure_seq_str)


def tokenize_directory(pdb_dir: Path, output_dir: Path, vocab_size: int, overwrite: bool) -> None:
    if not pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {pdb_dir}")

    print(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
    processor = PdbQuantizer(structure_vocab_size=vocab_size)

    for pdb_file in tqdm(pdb_files, desc="Tokenizing"):
        try:
            quantize_pdb(pdb_file, output_dir, processor, overwrite)
        except Exception as exc:
            print(f"Failed on {pdb_file.name}: {exc}")
            continue

    print(f"Done. Outputs in {output_dir}")


def main() -> None:
    args = parse_args()
    tokenize_directory(args.pdb_dir, args.output_dir, args.vocab_size, args.overwrite)


if __name__ == "__main__":
    main()