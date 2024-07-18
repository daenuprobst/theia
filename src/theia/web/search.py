import gzip
from shutil import copyfileobj
import pandas as pd
from flask import Blueprint, render_template, request, jsonify
from werkzeug.exceptions import abort

from drfp import DrfpEncoder
from annoy import AnnoyIndex

from theia import dm

bp = Blueprint("search", __name__)

annoy_index = None
rheadb = None
rhea_reactions = None


def init_search_db():
    global annoy_index
    global rheadb
    global rhea_reactions

    model_file = dm.new_path(f"rhea-drfp.ann")

    if not dm.exists(f"rhea-drfp.ann"):
        with gzip.open(dm.get_path(f"rhea-drfp.ann.gz"), "rb") as f_in:
            with open(model_file, "wb") as f_out:
                copyfileobj(f_in, f_out)

    annoy_index = AnnoyIndex(2048, metric="angular")
    annoy_index.load(str(model_file))

    rheadb = pd.read_csv(dm.get_path("rheadb.csv.gz")).fillna("-")
    rhea_reactions = pd.read_csv(dm.get_path("rhea-reactions.csv.gz"))
    rhea_reactions = rhea_reactions.set_index("RHEA_ID")


@bp.route("/")
def index():
    return render_template("search/index.html")


@bp.route("/search/knn", methods=["POST"])
def knn():
    params = request.get_json()
    k = int(params["k"])

    rxn_fp = DrfpEncoder.encode(
        [params["smiles"]],
        show_progress_bar=False,
        root_central_atom=False,
        radius=2,
        include_hydrogens=True,
        n_folded_length=2048,
    )[0]

    nn_ids = annoy_index.get_nns_by_vector(rxn_fp, k)
    data = [
        {"index": idx, "id": row.id, "ec": row.ec, "rxn": row.rxn}
        for idx, row in rheadb.loc[rheadb["id"].isin(nn_ids)].iterrows()
    ]

    for item in data:
        item["definition"] = rhea_reactions.at[item["id"], "DEFINITION"]
        item["equation"] = rhea_reactions.at[item["id"], "EQUATION"]

    return (
        jsonify({"success": True, "items": data}),
        200,
        {"ContentType": "application/json"},
    )


@bp.route("/search/info", methods=["POST"])
def info():
    params = request.get_json()
    item = rheadb.loc[params["index"]].to_dict()
    return (
        jsonify({"success": True, "info": item}),
        200,
        {"ContentType": "application/json"},
    )
