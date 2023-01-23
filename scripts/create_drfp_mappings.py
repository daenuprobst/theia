import pickle
import typer
from drfp import DrfpEncoder
import pandas as pd


def main(input_file: str, output_file: str, reaction_filed: str = "rxn"):
    df = pd.read_csv(input_file)

    # Replace ecreact style reactions with standard reaction SMILES
    df[reaction_filed] = df[reaction_filed].str.replace(r"\|.*>", ">>", regex=True)

    df["fps"], mapping = DrfpEncoder.encode(
        df[reaction_filed],
        mapping=True,
        show_progress_bar=True,
        root_central_atom=False,
        radius=2,
        include_hydrogens=True,
        n_folded_length=10240,
    )

    with open(output_file, "wb+") as f:
        pickle.dump(mapping, f)


if __name__ == "__main__":
    typer.run(main)
