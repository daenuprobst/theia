import re
import pickle
from pathlib import Path
from typing import List
from collections import defaultdict

import click
import pandas as pd
import numpy as np

from drfp import DrfpEncoder
from pycm import ConfusionMatrix
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as nnf

from rdkit.Chem import (
    rdMolDescriptors,
    rdChemReactions,
    MolFromSmiles,
    MolToSmiles,
)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import shap
import plotext as plt
from theia.ml import MLPClassifier, InferenceReactionDataset, ReactionDataset


def plot(train_loss, val_loss):
    plt.clear_figure()
    plt.clear_data()
    # plt.clear_color()

    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")

    # plt.title("Multiple Data Set")
    plt.show()


def predict(model, device, data_set, label_encoder, topk=10):
    data_sample = next(iter(DataLoader(data_set)))
    data_sample.to(device)
    pred_raw = model(data_sample)
    probs = torch.flatten(nnf.softmax(pred_raw, dim=1)).detach().numpy()
    pred = pred_raw.max(1, keepdim=True)[1]
    y_pred = torch.flatten(pred).tolist()

    topk_indices = (-probs).argsort()[:topk]
    probabilities = {
        label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probs)
    }

    return label_encoder.inverse_transform(y_pred)[0], probabilities, topk_indices


def get_accuracy(model, device, data_loader):
    correct = 0.0
    total = 0.0

    y = []
    y_pred = []

    for data, labels in data_loader:
        data, labels = data.to(device), labels.to(device)

        target = model(data)
        pred = target.max(1, keepdim=True)[1]  # get the index of the max logit
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += int(labels.shape[0])

        y.extend(torch.flatten(labels.view_as(pred)).tolist())
        y_pred.extend(torch.flatten(pred).tolist())

    return correct / total, y, y_pred


def get_cm(model, device, data_loader):
    correct = 0.0
    total = 0.0
    for data, labels in data_loader:
        data, labels = data.to(device), labels.to(device)

        target = model(data)
        pred = target.max(1, keepdim=True)[1]  # get the index of the max logit
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += int(labels.shape[0])

    return correct / total


def train(model, device, train_loader, valid_loader, optimizer, criterion, epoch):
    model.train()

    train_loss = 0.0
    for _, (data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, labels = data.to(device), labels.to(device)

        # Set gradients to zero
        optimizer.zero_grad()

        # Forward pass
        target = model(data)

        # Calculate the loss
        loss = criterion(target, labels)

        # Calculate the gradients
        loss.backward()

        # Update Weights
        optimizer.step()

        # Calculate the training loss
        train_loss += loss.item()

    # Validation
    valid_loss = 0.0
    model.eval()
    for data, labels in valid_loader:
        if torch.cuda.is_available():
            data, labels = data.to(device), labels.to(device)

        target = model(data)
        loss = criterion(target, labels)
        valid_loss += loss.item()

    train_loss = train_loss / len(train_loader)
    valid_loss = valid_loss / len(valid_loader)

    train_acc, _, _ = get_accuracy(model, device, train_loader)
    valid_acc, _, _ = get_accuracy(model, device, valid_loader)
    print(f"Epoch {epoch}: Training loss: {train_loss}, Validation loss: {valid_loss}.")
    print(
        f"Epoch {epoch}: Training accuracy: {train_acc}, Validation accuracy: {valid_acc}."
    )

    return (train_loss, valid_loss, train_acc, valid_acc)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        Path(Path(__file__).resolve().parent, "../data/ecreact-nofilter-1.0.csv")
    )
    df = df[~df.ec.str.contains("null")]
    df.rxn_smiles = df.rxn_smiles.apply(lambda x: re.sub(r"\|.*?\>\>", ">>", x))
    df["ec1"] = df.ec.str.split(".", expand=True)[0]
    df["ec2"] = df.ec.str.split(".", expand=True)[1]
    df["ec3"] = df.ec.str.split(".", expand=True)[2]
    df["ec12"] = df.ec1.astype(str) + "." + df.ec2.astype(str)
    df["ec123"] = df.ec12.astype(str) + "." + df.ec3.astype(str)

    # Remove transporters
    df = df[df.ec1 != "7"]

    return df


def calculate_cache_mqn(df) -> pd.DataFrame:
    cache_path = Path(__file__).resolve().parent
    cache_file = Path(cache_path, "../data/ecreact-dmqn.cache.pkl")
    if cache_file.exists():
        df_tmp = pickle.load(open(cache_file, "rb"))
        df["dmqns"] = df_tmp.dmqns
    else:
        dmqns = []
        for rxn_smiles in tqdm(df.rxn_smiles):
            rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles)
            dmqn = np.zeros(42)
            for reactant in rxn.GetReactants():
                reactant = MolFromSmiles(MolToSmiles(reactant))

                dmqn = np.add(dmqn, np.array(rdMolDescriptors.MQNs_(reactant)))

            for product in rxn.GetProducts():
                product = MolFromSmiles(MolToSmiles(product))
                dmqn = np.subtract(dmqn, np.array(rdMolDescriptors.MQNs_(product)))

            dmqn_pos = dmqn.clip(min=0)
            dmqn_neg = np.abs(dmqn.clip(max=0))
            dmqns.append(np.concatenate((dmqn_pos, dmqn_neg)))

        df["dmqns"] = dmqns
        pickle.dump(df, open(cache_file, "wb+"))

    return df


def calculate_cache_rxnfp(df) -> pd.DataFrame:
    cache_path = Path(__file__).resolve().parent
    cache_file = Path(cache_path, "../data/ecreact-rxnfp.cache.pkl")
    if cache_file.exists():
        df_tmp = pickle.load(open(cache_file, "rb"))
        df["rxnfps"] = df_tmp.rxnfps
    else:
        print("rxnfp not loaded!")

    return df


def get_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    return torch.device(dev)


def train_test_model(
    data_set_train,
    data_set_valid,
    data_set_test,
    label_encoder,
    device,
    input_dim: int = 10,
    hidden_dim: int = 1664,
    output_dim: int = 2,
    epochs: int = 10,
    patience: int = 5,
) -> ConfusionMatrix:
    model = MLPClassifier(input_dim, hidden_dim, output_dim)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    model.to(device)

    training_matrics = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
    }
    for epoch in range(epochs):
        train_loss, valid_loss, train_acc, valid_acc = train(
            model,
            device,
            DataLoader(data_set_train, batch_size=64),
            DataLoader(data_set_valid, batch_size=64),
            optimizer,
            criterion,
            epoch,
        )

        scheduler.step()

        # Early stopping based on past mean
        if len(training_matrics["valid_loss"]) >= patience:
            mean_past = np.mean(np.array(training_matrics["valid_loss"][-patience:]))
            if valid_loss - mean_past > -0.001:
                break

        training_matrics["train_loss"].append(train_loss)
        training_matrics["valid_loss"].append(valid_loss)
        training_matrics["train_acc"].append(train_acc)
        training_matrics["valid_acc"].append(valid_acc)

        print(training_matrics["valid_loss"])
        plot(training_matrics["train_loss"], training_matrics["valid_loss"])

    test_acc, y, y_pred = get_accuracy(
        model, device, DataLoader(data_set_test, batch_size=64)
    )

    y = label_encoder.inverse_transform(y)
    y_pred = label_encoder.inverse_transform(y_pred)

    cm = ConfusionMatrix(actual_vector=y, predict_vector=y_pred)

    print(f"Test accuracy: {test_acc}")
    print(cm.overall_stat)
    print(f"Accuracy : {cm.overall_stat['Overall ACC']:.3f}")
    print(f"MCC : {cm.overall_stat['Overall MCC']:.3f}")
    print(f"CEN : {cm.overall_stat['Overall CEN']:.3f}")

    return model, cm, training_matrics


def get_deep_explainer(model, data_set, device, sample_size=100):
    data, _ = next(iter(DataLoader(data_set, batch_size=sample_size, shuffle=True)))

    data = data.to(device)

    return shap.DeepExplainer(model, data)


def explain(data_set, explainer, label_encoder, probs, indices):
    rxns = data_set.rxns
    mappings = data_set.mappings
    shinglings = data_set.shinglings
    data_sample = next(iter(DataLoader(data_set)))

    shap_values = explainer.shap_values(data_sample)
    weights = shap_values[0][0]
    rxn = rxns[0]
    mapping = mappings[0]

    rxn = rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)

    reactants = [mol for mol in rxn.GetReactants()]
    for mol in reactants:
        mol.UpdatePropertyCache()

    products = [mol for mol in rxn.GetProducts()]
    for mol in products:
        mol.UpdatePropertyCache()

    reactant_highlights = [set() for _ in range(len(reactants))]
    product_highlights = [set() for _ in range(len(products))]

    reactant_weights = [defaultdict(int) for _ in range(len(reactants))]
    product_weights = [defaultdict(int) for _ in range(len(products))]

    for c in indices:
        weights = shap_values[c][0]
        for i in range(len(reactant_weights)):
            for j in range(reactants[i].GetNumHeavyAtoms()):
                reactant_weights[i][j] = 0.0

        for i in range(len(product_weights)):
            for j in range(products[i].GetNumHeavyAtoms()):
                product_weights[i][j] = 0.0

        for reactant_index, reactant in enumerate(mapping["reactants"]):
            for bit, atom_indices in reactant.items():
                for ai in atom_indices:
                    for indices in ai:
                        reactant_highlights[reactant_index].update(indices)
                        for idx in indices:
                            reactant_weights[reactant_index][idx] += weights[bit]

        for product_index, product in enumerate(mapping["products"]):
            for bit, atom_indices in product.items():
                for ai in atom_indices:
                    for indices in ai:
                        product_highlights[product_index].update(indices)
                        for idx in indices:
                            product_weights[product_index][idx] += weights[bit]

        svg = "<svg "
        svg += f'data-smiles="{rxns[0]}" '

        rw = ";".join(
            [
                ",".join(
                    map(
                        str,
                        [
                            reactant_weights[idx][i]
                            for i in range(len(reactant_weights[idx]))
                        ],
                    )
                )
                for idx in range(len(reactant_weights))
            ]
        )

        pw = ";".join(
            [
                ",".join(
                    map(
                        str,
                        [
                            product_weights[idx][i]
                            for i in range(len(product_weights[idx]))
                        ],
                    )
                )
                for idx in range(len(product_weights))
            ]
        )

        svg += f'data-smiles-reactant-weights="{rw}" '
        svg += f'data-smiles-product-weights="{pw}" '
        svg += "data-smiles-reaction-options=\"{ 'weights': { 'normalize': true } }\""
        svg += " />"

        label = label_encoder.inverse_transform([c])[0]
        # print(f"<h3>{label}</h3>")
        # print(f"<p>{probs[label] * 100}%</p>")
        # print(f"<p>{np.sum(weights)}</p>")
        # print(svg)


@click.command()
# @click.option("--variable", default="ec123", type=str)
# @click.option("--split", default=0.1, type=float)
@click.argument("train", type=str)
@click.argument("test", type=str)
@click.argument("valid", type=str)
@click.argument("output", type=str)
def main(train, valid, test, output):
    device = get_device()

    model_file = Path(f"{output}.pt")
    cm_file = Path(f"{output}.cm")
    training_metrics_file = Path(f"{output}.metrics.pkl")
    label_encoder_file = Path(f"{output}-le.pkl")
    explainer_background_file = Path(f"{output}-background.pkl")

    # df = load_data()
    # df = calculate_cache_drfp(df)
    # df = calculate_cache_mqn(df)
    # df = calculate_cache_rxnfp(df)

    df_train = pd.read_csv(train)
    df_valid = pd.read_csv(valid)
    df_test = pd.read_csv(test)

    df_train["fps"] = df_train.fps.apply(
        lambda x: np.array(list(map(int, x.split(";"))))
    )
    df_valid["fps"] = df_valid.fps.apply(
        lambda x: np.array(list(map(int, x.split(";"))))
    )
    df_test["fps"] = df_test.fps.apply(lambda x: np.array(list(map(int, x.split(";")))))

    df_train.label = df_train.label.astype(str)
    df_valid.label = df_valid.label.astype(str)
    df_test.label = df_test.label.astype(str)

    le = LabelEncoder()
    le.fit(pd.concat([df_train.label, df_valid.label, df_test.label]))

    df_train["label"] = le.transform(df_train.label)
    df_valid["label"] = le.transform(df_valid.label)
    df_test["label"] = le.transform(df_test.label)

    n_classes = len(le.classes_)

    input_dim = len(df_train["fps"].iloc[0])

    data_set_train = ReactionDataset(df_train)
    data_set_valid = ReactionDataset(df_valid)
    data_set_test = ReactionDataset(df_test)

    if not label_encoder_file.exists():
        with open(label_encoder_file, "wb") as f:
            pickle.dump(le, f)

    # if model_file.exists():
    #     model = MLPClassifier(input_dim, 1664, n_classes)
    #     model.load_state_dict(model_file)

    if not model_file.exists():
        model, cm, training_matrics = train_test_model(
            data_set_train,
            data_set_valid,
            data_set_test,
            le,
            device,
            input_dim=input_dim,
            hidden_dim=1664,
            output_dim=n_classes,
            epochs=50,
        )

        with open(cm_file, "wb") as f:
            pickle.dump(cm, f)

        with open(training_metrics_file, "wb") as f:
            pickle.dump(training_matrics, f)

        torch.save(model.state_dict(), model_file)

    if not explainer_background_file.exists():
        with open(explainer_background_file, "wb+") as f:
            pickle.dump(ReactionDataset(df_train.sample(n=250)), f)


if __name__ == "__main__":
    main()
