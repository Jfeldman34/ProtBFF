from pathlib import Path
from Bio.PDB import PDBParser
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse
import numpy as np
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STANDARD_AA = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate ProSST embeddings.")
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=repo_root / "data" / "optimized_tokens_2048",
        help="Tokenized FASTA directory.",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=repo_root / "data" / "optimized",
        help="Source PDB directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "optimized_embeddings_2048",
        help="Output directory for .npy files.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="AI4Protein/ProSST-2048",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Hidden layer (-1 = last).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force device (default: cuda if available).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing outputs.",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_chain_center(chain):
    coords = []
    for residue in chain:
        if residue.id[0] == " ":
            for atom in residue:
                coords.append(atom.get_coord())
    if not coords:
        return None
    return np.mean(coords, axis=0)


def reassign_empty_chain_atoms(structure):
    valid_chains = [chain for chain in structure.get_chains() if chain.id != " "]
    if not valid_chains:
        return structure

    chain_centers = {}
    for chain in valid_chains:
        center = get_chain_center(chain)
        if center is not None:
            chain_centers[chain.id] = center

    empty_chain = None
    for chain in structure.get_chains():
        if chain.id == " ":
            empty_chain = chain
            break

    if empty_chain is None or not chain_centers:
        return structure

    for residue in empty_chain:
        if residue.id[0] == " ":
            residue_center = np.mean([atom.get_coord() for atom in residue], axis=0)
            nearest_chain_id = min(
                chain_centers,
                key=lambda chain_id: np.linalg.norm(residue_center - chain_centers[chain_id]),
            )
            target_chain = structure[0][nearest_chain_id]
            empty_chain.detach_child(residue.id)
            target_chain.add(residue)

    return structure


def parse_pdb_sequence(pdb_file: Path) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    structure = reassign_empty_chain_atoms(structure)

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    aa = STANDARD_AA.get(residue.get_resname())
                    if aa:
                        residues.append(aa)
    return "".join(residues)


def read_structure_sequence(fasta_file: Path) -> list[int]:
    with open(fasta_file, "r") as handle:
        lines = handle.readlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid FASTA file format: {fasta_file}")
    return [int(i) for i in lines[1].strip().split(",") if i]


def tokenize_structure_sequence(structure_sequence: list[int]) -> torch.Tensor:
    shift_seq = [1] + [i + 3 for i in structure_sequence] + [2]
    return torch.tensor([shift_seq], dtype=torch.long, device=device)


def clean_sequence(sequence: str) -> str:
    return "".join([aa for aa in sequence if aa in STANDARD_AA.values()])


@torch.no_grad()
def get_per_residue_embeddings(
    prosst_model,
    tokenizer,
    sequence: str,
    structure_sequence: list[int],
    layer: int,
):
    cleaned_sequence = clean_sequence(sequence)

    if len(cleaned_sequence) != len(structure_sequence):
        min_len = min(len(cleaned_sequence), len(structure_sequence))
        print(
            f"Length mismatch (seq={len(cleaned_sequence)}, struct={len(structure_sequence)}); "
            f"trimming to {min_len}"
        )
        cleaned_sequence = cleaned_sequence[:min_len]
        structure_sequence = structure_sequence[:min_len]

    tokenized = tokenizer(
        [cleaned_sequence],
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
    ).to(device)

    ss_tensor = tokenize_structure_sequence(structure_sequence)

    outputs = prosst_model(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
        ss_input_ids=ss_tensor,
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states
    layer_index = layer if layer >= 0 else len(hidden_states) + layer
    if layer_index < 0 or layer_index >= len(hidden_states):
        raise ValueError(
            f"Layer {layer} out of range for model with {len(hidden_states)} hidden states."
        )

    embeddings = hidden_states[layer_index][0][1:-1].cpu().numpy()
    return embeddings


def infer_pdb_id(fasta_stem: str) -> str:
    return fasta_stem[:-5] if fasta_stem.endswith("_full") else fasta_stem


def process_fasta_file(
    fasta_file: Path,
    pdb_dir: Path,
    output_dir: Path,
    prosst_model,
    tokenizer,
    layer: int,
    overwrite: bool,
):
    pdb_id = infer_pdb_id(fasta_file.stem)
    pdb_file = pdb_dir / f"{pdb_id}.pdb"
    output_path = output_dir / f"{fasta_file.stem}_embeddings.npy"

    if output_path.exists() and not overwrite:
        print(f"Skipping {fasta_file.name}")
        return

    if not pdb_file.exists():
        print(f"Skipping {fasta_file.name} - missing PDB")
        return

    print(f"Processing {fasta_file.name}")
    sequence = parse_pdb_sequence(pdb_file)
    structure_sequence = read_structure_sequence(fasta_file)
    embeddings = get_per_residue_embeddings(
        prosst_model, tokenizer, sequence, structure_sequence, layer
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved {output_path.name} (shape={embeddings.shape})")


def main() -> None:
    global device
    args = parse_args()
    device = resolve_device(args.device)

    if not args.token_dir.exists():
        raise FileNotFoundError(f"Token directory not found: {args.token_dir}")
    if not args.pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {args.pdb_dir}")

    fasta_files = sorted(args.token_dir.glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {args.token_dir}")

    print(f"Device: {device}")
    print(f"Loading {args.model_name}...")
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name, trust_remote_code=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    for fasta_file in fasta_files:
        try:
            process_fasta_file(
                fasta_file,
                args.pdb_dir,
                args.output_dir,
                model,
                tokenizer,
                args.layer,
                args.overwrite,
            )
        except Exception as exc:
            print(f"Failed on {fasta_file.name}: {exc}")
            continue

    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()

