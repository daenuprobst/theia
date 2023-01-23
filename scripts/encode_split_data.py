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

    df["fps"] = DrfpEncoder.encode(
        df.rxn,
        show_progress_bar=True,
        root_central_atom=False,
        radius=2,
        include_hydrogens=True,
        n_folded_length=10240,
    )

    for ec in ["ec1", "ec12", "ec123"]:
        X = df.rxn.to_numpy()
        y = df[ec].to_numpy()
        fps = df.fps.to_numpy()
        groups = df.ec_1.to_numpy()

        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        sss_valid = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

        for i, (train_index, test_valid_index) in enumerate(sss.split(X, groups)):
            for _, (test_index, valid_index) in enumerate(
                sss_valid.split(
                    X[test_valid_index],
                    groups[test_valid_index],
                )
            ):
                X_train = X[train_index]
                y_train = y[train_index]
                fps_train = fps[train_index]

                X_valid = X[valid_index]
                y_valid = y[valid_index]
                fps_valid = fps[valid_index]

                X_test = X[test_index]
                y_test = y[test_index]
                fps_test = fps[test_index]

                df_train = pd.DataFrame(
                    {
                        "rxn_smiles": X_train,
                        "label": y_train,
                        "fps": [";".join(map(str, fp)) for fp in fps_train],
                    }
                )
                df_valid = pd.DataFrame(
                    {
                        "rxn_smiles": X_valid,
                        "label": y_valid,
                        "fps": [";".join(map(str, fp)) for fp in fps_valid],
                    }
                )
                df_test = pd.DataFrame(
                    {
                        "rxn_smiles": X_test,
                        "label": y_test,
                        "fps": [";".join(map(str, fp)) for fp in fps_test],
                    }
                )

                df_train.to_csv(f"{output_path}-{i}-{ec}-train.csv", index=False)
                df_valid.to_csv(f"{output_path}-{i}-{ec}-valid.csv", index=False)
                df_test.to_csv(f"{output_path}-{i}-{ec}-test.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
