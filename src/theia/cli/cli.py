from pathlib import Path
from warnings import filterwarnings

filterwarnings("ignore")
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from theia.api import predict, predict_all
from theia import download


def download_internal():
    download()


def predict_internal(
    model: str = typer.Argument(
        ...,
        help="The name of the model. Options are rheadb.ec1, rheadb.ec12, rheadb.ec123, ecreact.ec1, ecreact.ec12, and ecreact.ec123.",
    ),
    rxn_smiles: str = typer.Argument(
        ..., help="The reaction smiles in the form a.b>>c.d."
    ),
    topk: Optional[int] = 5,
    probs: Optional[bool] = False,
    pretty: Optional[bool] = False,
):
    is_path = False

    try:
        if Path(rxn_smiles).exists():
            is_path = True
    except:
        ...  # Not raising or warning as exception will be thrown through malformed smiles

    if is_path:
        with open(rxn_smiles, "r") as f:
            for val in predict_all(model, f, topk, False, probs):
                if not probs:
                    print(val)
                else:
                    print(";".join([f"{key},{value}" for key, value in val[1].items()]))

        return

    if "," not in rxn_smiles:
        val = predict(model, rxn_smiles, topk, False, probs)

        if not probs:
            print(val)
            return

        if pretty:
            table = Table()
            table.add_column("Prediction")
            table.add_column("Probability [%]", style="magenta", justify="right")

            for key, value in val[1].items():
                table.add_row(
                    key,
                    str(round(value * 100, 2)),
                )

            console = Console()
            console.print(table)
        else:
            print(";".join([f"{key},{value}" for key, value in val[1].items()]))
    else:
        vals = predict_all(model, rxn_smiles.split(","), topk, False, probs)

        for val in vals:
            if not probs:
                print(val)
            else:
                if pretty:
                    table = Table(title="Star Wars Movies")
                    table.add_column("Prediction")
                    table.add_column(
                        "Probability [%]", style="magenta", justify="right"
                    )

                    for key, value in val[1].items():
                        table.add_row(
                            key,
                            round(value * 100, 2),
                        )

                    console = Console()
                    console.print(table)
                else:
                    print(";".join([f"{key},{value}" for key, value in val[1].items()]))


def predict_cli():
    typer.run(predict_internal)
