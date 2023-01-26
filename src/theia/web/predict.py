from pathlib import Path
from operator import attrgetter
import numpy as np
from flask import (
    Blueprint,
    request,
    jsonify,
)

from drfp import DrfpEncoder

from theia.web.helpers import (
    load_models,
    get_device,
    get_deep_explainer,
    predict,
    explain,
)
from theia.ml import InferenceReactionDataset


bp = Blueprint("predict", __name__)

models = None
device = get_device()
explainer_cache = {}

def init_models():
    global models
    models = {"rheadb": load_models("rheadb"), "ecreact": load_models("ecreact")}


@bp.route("/predict/ec", methods=["POST"])
def ec():
    params = request.get_json()
    source = "rheadb"

    if "source" in params:
        source = params["source"]

    m = params["model"]
    smiles = params["smiles"]

    model, label_encoder, background, drfp_map = models[source][m]
    model = model.to(device)
    dataset = InferenceReactionDataset([smiles])

    pred, probs, topk_indices = predict(model, device, dataset, label_encoder, 5)

    explainer = None

    if f"{source}.{m}" in explainer_cache:
        explainer = explainer_cache[f"{source}.{m}"]
    else:
        explainer = get_deep_explainer(model, background, device)
        explainer_cache[f"{source}.{m}"] = explainer

    explained_reactions = explain(
        dataset, explainer, label_encoder, probs, topk_indices, drfp_map
    )

    explain_values = {
        e.label: {
            "reactant_weights": e.reactant_weights,
            "product_weights": e.product_weights,
            "neg_reactant_weights": e.neg_reactant_weights,
            "neg_product_weights": e.neg_product_weights,
            "pos_reactant_weights": e.pos_reactant_weights,
            "pos_product_weights": e.pos_product_weights,
            "total_weights": e.summed_weights(),
            "abs_neg_pos": e.abs_neg_pos(),
            "reactant_fragments": e.reactant_fragments,
            "product_fragments": e.product_fragments,
            "top_missing_fragments": e.top_missing_fragments,
        }
        for e in explained_reactions
    }

    json = jsonify(
        {
            "success": True,
            "pred": [
                {
                    "rxn": smiles,
                    "ec": ec,
                    "prob": float(prob),
                    "explain": explain_values[ec],
                }
                for ec, prob in sorted(
                    probs.items(), key=lambda item: item[1], reverse=True
                )
                if ec in explain_values
            ],
        }
    )

    return (
        json,
        200,
        {"ContentType": "application/json"},
    )
