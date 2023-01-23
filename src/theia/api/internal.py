from warnings import filterwarnings

filterwarnings("ignore")
from typing import Optional, List, Union

from theia.web.helpers import (
    load_models,
    get_device,
    get_deep_explainer,
)
from theia.web.helpers import explain as explain_internal
from theia.web.helpers import predict as predict_internal

from theia.ml import InferenceReactionDataset


def predict_one(
    rxn, model, label_encoder, background, drfp_map, device, probs, topk, explain
):
    dataset = InferenceReactionDataset([rxn])
    pred, probabilities, topk_indices = predict_internal(
        model, device, dataset, label_encoder, topk
    )

    result = [pred]

    if probs:
        top_k_classes = [label_encoder.inverse_transform([i])[0] for i in topk_indices]
        result.append({c: probabilities[c] for c in top_k_classes})

    if explain:
        explainer = get_deep_explainer(model, background, device)
        expls = explain_internal(
            dataset, explainer, label_encoder, probabilities, topk_indices, drfp_map
        )

        result.append(expls)

    return result


def predict_all(
    model_id: str,
    rxn_smiles: List[str],
    topk: Optional[int] = 5,
    explain: Optional[bool] = False,
    probs: Optional[bool] = False,
):
    vals = model_id.split(".")
    models = load_models(vals[0], [vals[1]])

    device = get_device()
    model, label_encoder, background, drfp_map = models[vals[1]]
    model = model.to(device)

    for rxn in rxn_smiles:
        result = predict_one(
            rxn.strip(),
            model,
            label_encoder,
            background,
            drfp_map,
            device,
            probs,
            topk,
            explain,
        )

        if len(result) == 1:
            yield result[0]
        else:
            yield tuple(result)


def predict(
    model_id: str,
    rxn_smiles: str,
    topk: Optional[int] = 5,
    explain: Optional[bool] = False,
    probs: Optional[bool] = False,
):
    vals = model_id.split(".")
    models = load_models(vals[0], [vals[1]])

    device = get_device()
    model, label_encoder, background, drfp_map = models[vals[1]]
    model = model.to(device)

    result = predict_one(
        rxn_smiles,
        model,
        label_encoder,
        background,
        drfp_map,
        device,
        probs,
        topk,
        explain,
    )

    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)
