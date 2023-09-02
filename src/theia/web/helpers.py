# Disable the "IPython could not be loaded!" warning
import pickle
from typing import Tuple, List, Dict, Set
from collections import defaultdict
from pathlib import Path

import numpy as np

from shap import Explainer, DeepExplainer, GradientExplainer, KernelExplainer

from torch import flatten, device, cuda, nn, from_numpy
from torch import load as load_module
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional, Module
from torch.autograd import Variable

from sklearn.preprocessing import LabelEncoder

from rdkit.Chem import rdChemReactions
from rdkit.Chem import AllChem

from theia import dm
from theia.ml import ExplainedReaction, MLPClassifier, ReactionDataset


def get_device():
    if cuda.is_available():
        return device("cuda:0")

    return device("cpu")


def load_models(
    source: str = "rheadb", names: List[str] = None, split: str = "0"
) -> Dict[str, Tuple[MLPClassifier, LabelEncoder, Dataset]]:
    if names is None:
        names = ["ec1", "ec12", "ec123"]

    models = {}

    for name in names:
        model_path = dm.get_path(f"{source}-{split}-{name}.pt")
        if not model_path.exists():
            continue

        classifier = None
        label_encoder = None
        background = None
        drfp_map = None

        with open(dm.get_path(f"{source}-map.pkl"), "rb") as f:
            drfp_map = pickle.load(f)

        with open(dm.get_path(f"{source}-{split}-{name}-le.pkl"), "rb") as f:
            label_encoder: LabelEncoder = pickle.load(f)

        with open(dm.get_path(f"{source}-{split}-{name}-background.pkl"), "rb") as f:
            background: ReactionDataset = pickle.load(f)

        classifier = MLPClassifier(10240, 1664, len(label_encoder.classes_))
        classifier.load_state_dict(load_module(model_path))
        classifier.eval()

        models[name] = (classifier, label_encoder, background, drfp_map)

    return models


def predict(
    model: Module,
    device: device,
    data_set: Dataset,
    label_encoder: LabelEncoder,
    topk: int = 10,
) -> Tuple[str, Dict[str, float], List[int]]:
    data_sample = next(iter(DataLoader(data_set)))
    data_sample = data_sample.to(device)
    pred_raw = model(data_sample)
    probs = flatten(functional.softmax(pred_raw, dim=1)).cpu().detach().numpy()
    pred = pred_raw.max(1, keepdim=True)[1]
    y_pred = flatten(pred).tolist()

    topk_indices = (-probs).argsort()[:topk]
    probabilities = {
        label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probs)
    }

    return label_encoder.inverse_transform(y_pred)[0], probabilities, topk_indices


def get_deep_explainer(
    model: Module, data_set: Dataset, device: device, sample_size: int = 100
) -> DeepExplainer:
    data, _ = next(iter(DataLoader(data_set, batch_size=sample_size, shuffle=True)))
    data = data.to(device)

    return DeepExplainer(model, data)


def get_gradient_explainer(
    model: Module, data_set: Dataset, device: device, sample_size: int = 100
) -> GradientExplainer:
    data, _ = next(iter(DataLoader(data_set, batch_size=sample_size, shuffle=True)))
    data = data.to(device)

    return GradientExplainer(model, data)


# Adapted from: https://stackoverflow.com/questions/70510341/shap-values-with-pytorch-kernelexplainer-vs-deepexplainer
def get_kernel_explainer(
    model: Module, data_set: Dataset, device: device, sample_size: int = 100
) -> KernelExplainer:

    f = lambda x: model(Variable(from_numpy(x))).detach().numpy()
    data, _ = next(iter(DataLoader(data_set, batch_size=sample_size, shuffle=True)))
    data = data.numpy()

    return KernelExplainer(f, data)


def add_hydrogens(smiles):
    smiles_with_hs = ""
    rxn = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)

    for mol in rxn.GetReactants():
        mol.UpdatePropertyCache()
        mol = AllChem.AddHs(mol)
        smiles_with_hs += (
            AllChem.MolToSmiles(mol, canonical=True, allHsExplicit=True) + "."
        )

    smiles_with_hs = smiles_with_hs.strip(".")
    smiles_with_hs += ">>"

    for mol in rxn.GetProducts():
        mol.UpdatePropertyCache()
        mol = AllChem.AddHs(mol)
        smiles_with_hs += (
            AllChem.MolToSmiles(mol, canonical=True, allHsExplicit=True) + "."
        )

    smiles_with_hs = smiles_with_hs.strip(".")
    return smiles_with_hs


def merge_hydrogens(weights, molecules):
    new_weights = []
    for i, mol in enumerate(molecules):
        mol.UpdatePropertyCache()
        mol = AllChem.AddHs(mol)
        mol = reindex_mol(mol)

        # Map hydrogens to the attached havy atom
        hydrogen_map = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetDegree() > 0:
                heavy_atom_idx = atom.GetNeighbors()[0].GetIdx()
                hydrogen_map[atom.GetIdx()] = heavy_atom_idx

        # Merge hydrogens
        new_mol_weights = defaultdict(float)
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx in hydrogen_map:
                new_mol_weights[hydrogen_map[idx]] += weights[i][idx]
            else:
                new_mol_weights[idx] += weights[i][idx]

        new_weights.append(new_mol_weights)

    return new_weights


def reindex_mol(mol):
    smiles = AllChem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
    return AllChem.MolFromSmiles(smiles)


def get_top_missing_fragments(fp, weights, drfp_map, k: int = 10):
    weights_abs = np.abs(weights)
    on_bits = np.where(fp == 1)[0]
    np.put(weights_abs, on_bits, [-1.0])

    topk_indices = (-weights_abs).argsort()[:k]

    result = []
    for idx, w in zip(topk_indices, weights[topk_indices]):
        result.append((list(drfp_map[idx]), w))

    return result


def explain(
    data_set: Dataset,
    explainer: Explainer,
    label_encoder: LabelEncoder,
    probs: Dict[str, float],
    indices: List[int],
    drfp_map: Dict[int, Set[str]],
) -> List[ExplainedReaction]:
    rxns = data_set.rxns
    mappings = data_set.mappings
    shingling = data_set.shinglings
    fps = data_set.X
    data_sample = next(iter(DataLoader(data_set)))

    shap_params = {}
    if isinstance(explainer, KernelExplainer):
        data_sample = data_sample.numpy()
        shap_params["silent"] = True

    shap_values = explainer.shap_values(data_sample, *shap_params)
    weights = shap_values[0][0]
    rxn = rxns[0]
    mapping = mappings[0]
    fp = fps[0].numpy()

    rxn = rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)

    reactants = [mol for mol in rxn.GetReactants()]
    for mol in reactants:
        mol.UpdatePropertyCache()
        mol = AllChem.AddHs(mol)

    products = [mol for mol in rxn.GetProducts()]
    for mol in products:
        mol.UpdatePropertyCache()
        mol = AllChem.AddHs(mol)

    reactant_highlights = [set() for _ in range(len(reactants))]
    product_highlights = [set() for _ in range(len(products))]

    reactant_weights = [defaultdict(float) for _ in range(len(reactants))]
    product_weights = [defaultdict(float) for _ in range(len(products))]

    neg_reactant_weights = [defaultdict(float) for _ in range(len(reactants))]
    neg_product_weights = [defaultdict(float) for _ in range(len(products))]

    pos_reactant_weights = [defaultdict(float) for _ in range(len(reactants))]
    pos_product_weights = [defaultdict(float) for _ in range(len(products))]

    results = []
    for c in indices:
        weights = shap_values[c][0]
        top_k_missing = get_top_missing_fragments(fp, weights, drfp_map)

        reactant_fragments = {}
        product_fragments = {}

        for i in range(len(reactant_weights)):
            for j in range(reactants[i].GetNumHeavyAtoms()):
                reactant_weights[i][j] = 0.0
                neg_reactant_weights[i][j] = 0.0
                pos_reactant_weights[i][j] = 0.0

        for i in range(len(product_weights)):
            for j in range(products[i].GetNumHeavyAtoms()):
                product_weights[i][j] = 0.0
                neg_product_weights[i][j] = 0.0
                pos_product_weights[i][j] = 0.0

        for reactant_index, reactant in enumerate(mapping["reactants"]):
            for bit, atom_indices in reactant.items():
                for ais in atom_indices:
                    for ai in ais:
                        reactant_highlights[reactant_index].update(ai)
                        for idx in ai:
                            weigth = weights[bit]

                            reactant_fragments[int(bit)] = (
                                list(shingling[bit]),
                                float(weigth),
                            )

                            reactant_weights[reactant_index][idx] += weigth

                            if weigth < 0.0:
                                neg_reactant_weights[reactant_index][idx] += weigth
                            else:
                                pos_reactant_weights[reactant_index][idx] += weigth

        for product_index, product in enumerate(mapping["products"]):
            for bit, atom_indices in product.items():
                for ais in atom_indices:
                    for ai in ais:
                        product_highlights[product_index].update(ai)
                        for idx in ai:
                            weight = weights[bit]

                            product_fragments[int(bit)] = (
                                list(shingling[bit]),
                                float(weight),
                            )

                            product_weights[product_index][idx] += weight

                            if weight < 0.0:
                                neg_product_weights[product_index][idx] += weight
                            else:
                                pos_product_weights[product_index][idx] += weight

        reactant_weights = merge_hydrogens(reactant_weights, reactants)
        product_weights = merge_hydrogens(product_weights, products)
        neg_reactant_weights = merge_hydrogens(neg_reactant_weights, reactants)
        neg_product_weights = merge_hydrogens(neg_product_weights, products)
        pos_reactant_weights = merge_hydrogens(pos_reactant_weights, reactants)
        pos_product_weights = merge_hydrogens(pos_product_weights, products)

        label = label_encoder.inverse_transform([c])[0]

        explained_reaction = ExplainedReaction(
            rxns[0],
            label,
            weights,
            [list(w.values()) for w in reactant_weights],
            [list(w.values()) for w in product_weights],
            [list(w.values()) for w in neg_reactant_weights],
            [list(w.values()) for w in neg_product_weights],
            [list(w.values()) for w in pos_reactant_weights],
            [list(w.values()) for w in pos_product_weights],
            list(reactant_fragments.values()),
            list(product_fragments.values()),
            top_k_missing,
            probs[label],
        )
        explained_reaction.remove_explicit_hydrogens()
        results.append(explained_reaction)

    return results

def explain_regression(
    data_set: Dataset,
    explainer: Explainer,
    drfp_map: Dict[int, Set[str]],
) -> ExplainedReaction:
    rxns = data_set.rxns
    mappings = data_set.mappings
    shingling = data_set.shinglings
    fps = data_set.X
    data_sample = next(iter(DataLoader(data_set)))

    shap_params = {}
    if isinstance(explainer, KernelExplainer):
        data_sample = data_sample.numpy()
        shap_params["silent"] = True

    shap_values = explainer.shap_values(data_sample, *shap_params)
    weights = shap_values[0][0]
    rxn = rxns[0]
    mapping = mappings[0]
    fp = fps[0].numpy()

    rxn = rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)

    reactants = [mol for mol in rxn.GetReactants()]
    for mol in reactants:
        mol.UpdatePropertyCache()
        mol = AllChem.AddHs(mol)

    products = [mol for mol in rxn.GetProducts()]
    for mol in products:
        mol.UpdatePropertyCache()
        mol = AllChem.AddHs(mol)

    reactant_highlights = [set() for _ in range(len(reactants))]
    product_highlights = [set() for _ in range(len(products))]

    reactant_weights = [defaultdict(float) for _ in range(len(reactants))]
    product_weights = [defaultdict(float) for _ in range(len(products))]

    neg_reactant_weights = [defaultdict(float) for _ in range(len(reactants))]
    neg_product_weights = [defaultdict(float) for _ in range(len(products))]

    pos_reactant_weights = [defaultdict(float) for _ in range(len(reactants))]
    pos_product_weights = [defaultdict(float) for _ in range(len(products))]

    weights = shap_values[0]
    top_k_missing = get_top_missing_fragments(fp, weights, drfp_map)

    reactant_fragments = {}
    product_fragments = {}

    for i in range(len(reactant_weights)):
        for j in range(reactants[i].GetNumHeavyAtoms()):
            reactant_weights[i][j] = 0.0
            neg_reactant_weights[i][j] = 0.0
            pos_reactant_weights[i][j] = 0.0

    for i in range(len(product_weights)):
        for j in range(products[i].GetNumHeavyAtoms()):
            product_weights[i][j] = 0.0
            neg_product_weights[i][j] = 0.0
            pos_product_weights[i][j] = 0.0

    for reactant_index, reactant in enumerate(mapping["reactants"]):
        for bit, atom_indices in reactant.items():
            for ais in atom_indices:
                for ai in ais:
                    reactant_highlights[reactant_index].update(ai)
                    for idx in ai:
                        weigth = weights[bit]

                        reactant_fragments[int(bit)] = (
                            list(shingling[bit]),
                            float(weigth),
                        )

                        reactant_weights[reactant_index][idx] += weigth

                        if weigth < 0.0:
                            neg_reactant_weights[reactant_index][idx] += weigth
                        else:
                            pos_reactant_weights[reactant_index][idx] += weigth

    for product_index, product in enumerate(mapping["products"]):
        for bit, atom_indices in product.items():
            for ais in atom_indices:
                for ai in ais:
                    product_highlights[product_index].update(ai)
                    for idx in ai:
                        weight = weights[bit]

                        product_fragments[int(bit)] = (
                            list(shingling[bit]),
                            float(weight),
                        )

                        product_weights[product_index][idx] += weight

                        if weight < 0.0:
                            neg_product_weights[product_index][idx] += weight
                        else:
                            pos_product_weights[product_index][idx] += weight

    reactant_weights = merge_hydrogens(reactant_weights, reactants)
    product_weights = merge_hydrogens(product_weights, products)
    neg_reactant_weights = merge_hydrogens(neg_reactant_weights, reactants)
    neg_product_weights = merge_hydrogens(neg_product_weights, products)
    pos_reactant_weights = merge_hydrogens(pos_reactant_weights, reactants)
    pos_product_weights = merge_hydrogens(pos_product_weights, products)

    explained_reaction = ExplainedReaction(
        rxns[0],
        "",
        weights,
        [list(w.values()) for w in reactant_weights],
        [list(w.values()) for w in product_weights],
        [list(w.values()) for w in neg_reactant_weights],
        [list(w.values()) for w in neg_product_weights],
        [list(w.values()) for w in pos_reactant_weights],
        [list(w.values()) for w in pos_product_weights],
        list(reactant_fragments.values()),
        list(product_fragments.values()),
        top_k_missing,
        0.0,
    )

    explained_reaction.remove_explicit_hydrogens()

    return explained_reaction