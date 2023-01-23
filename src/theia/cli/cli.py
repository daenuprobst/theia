from warnings import filterwarnings

filterwarnings("ignore")
from typing import Optional
import typer

from theia.web.helpers import (
    load_models,
    get_device,
    predict,
    get_deep_explainer,
)

from theia.web.helpers import explain as explain_internal
from theia.ml import InferenceReactionDataset


def predict_internal(
    model: str = typer.Argument(help="Test for help"),
    rxn_smiles: str = typer.Argument(help="Test for help"),
    topk: Optional[int] = 5,
    explain: Optional[bool] = False,
    probs: Optional[bool] = False,
):
    vals = model.split(".")
    models = load_models(vals[0], [vals[1]])

    model, label_encoder, background, drfp_map = models[vals[1]]
    dataset = InferenceReactionDataset([rxn_smiles])
    device = get_device()

    pred, probabilities, topk_indices = predict(
        model, device, dataset, label_encoder, topk
    )

    if probs:
        top_k_classes = [label_encoder.inverse_transform([i])[0] for i in topk_indices]
        print(",".join([f"{c}:{probabilities[c]}" for c in top_k_classes]))

    if explain:
        explainer = get_deep_explainer(model, background, device)
        explained_reactions = explain_internal(
            dataset, explainer, label_encoder, probabilities, topk_indices, drfp_map
        )

        print(explained_reactions)


def predict_cli():
    typer.run(predict_internal)
