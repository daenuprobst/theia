from pathlib import Path
from warnings import filterwarnings

filterwarnings("ignore")
from typing import Optional
import typer
from theia.api import predict, predict_all


def predict_internal(
    model: str = typer.Argument(..., help="Test for help"),
    rxn_smiles: str = typer.Argument(..., help="Test for help"),
    topk: Optional[int] = 5,
    probs: Optional[bool] = False,
):
    if Path(rxn_smiles).exists():
        with open(rxn_smiles, "r") as f:
            for val in predict_all(model, f, topk, False, probs):
                if not probs:
                    print(val)
                else:
                    print(";".join([f"{key},{value}" for key, value in val[1].items()]))

        return

    if "," not in rxn_smiles:
        vals = predict(model, rxn_smiles, topk, False, probs)

        if not probs:
            print(vals)
            return

        print(";".join([f"{key},{value}" for key, value in vals[1].items()]))
    else:
        vals = predict_all(model, rxn_smiles.split(","), topk, False, probs)

        for val in vals:
            if not probs:
                print(val)
            else:
                print(";".join([f"{key},{value}" for key, value in val[1].items()]))


def predict_cli():
    typer.run(predict_internal)
