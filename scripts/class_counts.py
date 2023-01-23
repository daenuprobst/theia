from pathlib import Path
import click
import typer
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from drfp import DrfpEncoder


def main(path: str, output_path: str):
    df = pd.read_csv(path)

    # Replace ecreact style reactions with standard reaction SMILES
    df["rxn"] = df.rxn.str.replace(r"\|.*>", ">>", regex=True)

    df_nan = df[~df.ec.notna()]
    df = df[df.ec.notna()]

    df[["ec_1", "ec_2", "ec_3", "ec_4"]] = df.ec.str.split(".", expand=True)

    df["ec1"] = df.ec_1.astype(str)
    df["ec12"] = df.ec_1.astype(str) + "." + df.ec_2.astype(str)
    df["ec123"] = (
        df.ec_1.astype(str) + "." + df.ec_2.astype(str) + "." + df.ec_3.astype(str)
    )

    df = df[df.ec1 != "7"]

    data = {"level": [], "ec": [], "count": []}
    for ec, count in df["ec1"].value_counts().items():
        data["level"].append(1)
        data["ec"].append(ec)
        data["count"].append(count)

    for ec, count in df["ec12"].value_counts().items():
        data["level"].append(2)
        data["ec"].append(ec)
        data["count"].append(count)

    for ec, count in df["ec123"].value_counts().items():
        data["level"].append(3)
        data["ec"].append(ec)
        data["count"].append(count)

    df_out = pd.DataFrame(data)
    df_out.to_csv(output_path, index=None)


if __name__ == "__main__":
    typer.run(main)
