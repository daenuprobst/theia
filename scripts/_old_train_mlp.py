import re
import gzip
import pickle
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
import numpy as np

from drfp import DrfpEncoder
from pycm import ConfusionMatrix
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

import lightgbm as lgb

from rdkit.Chem import (
    rdMolDescriptors,
    rdChemReactions,
    AddHs,
    MolFromSmiles,
    MolToSmiles,
    Descriptors,
    Descriptors3D,
)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_mlp_model(
    train_X: np.array,
    train_y: np.array,
    eval_X: np.array,
    n_classes: int,
    neurons: int = 1664,
    epochs: int = 10,
) -> tuple[list, keras.models.Sequential]:
    """
    Get predictaions using a simple MLP.
    """

    model = keras.models.Sequential(
        [
            keras.Input((len(train_X[0]),)),
            keras.layers.Dense(neurons, activation=tf.nn.tanh),
            keras.layers.Dense(n_classes, activation=tf.nn.softmax),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

    model.fit(train_X, train_y, epochs=epochs, batch_size=64, validation_split=0.1)

    return (np.argmax(model.predict(eval_X), axis=-1), model)


def get_multimodal_mlp_model(
    train_X: Tuple[np.array], train_y: np.array, eval_X: Tuple[np.array], n_classes: int
) -> tuple[list, keras.models.Sequential]:
    """
    Get predictaions using a multimodal MLP.
    """

    # define two sets of inputs
    inputA = keras.Input(shape=(len(train_X[0][0]),))
    inputB = keras.Input(shape=(len(train_X[1][0]),))

    # the first branch operates on the first input
    x = keras.layers.Dense(1664, activation=tf.nn.tanh)(inputA)
    x = keras.models.Model(inputs=inputA, outputs=x)

    # the second branch operates on the second input
    y = keras.layers.Dense(16, activation=tf.nn.tanh)(inputB)
    y = keras.models.Model(inputs=inputB, outputs=y)

    # combine the output of the two branches
    combined = keras.layers.concatenate([x.output, y.output])
    z = keras.layers.Dense(n_classes, activation=tf.nn.softmax)(combined)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = keras.models.Model(inputs=[x.input, y.input], outputs=z)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

    model.fit(train_X, train_y, epochs=10, batch_size=64, validation_split=0.1)

    return (np.argmax(model.predict(eval_X), axis=-1), model)


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

    return df


def calculate_cache_drfp(df) -> pd.DataFrame:
    cache_path = Path(__file__).resolve().parent
    cache_file = Path(cache_path, "../data/ecreact-drfp.cache.pkl")
    map_file = Path(cache_path, "../data/ecreact-drfp-map.cache.pkl")
    if cache_file.exists():
        df_tmp = pickle.load(open(cache_file, "rb"))
        df["fps"] = df_tmp.fps
    else:
        df["fps"], mapping = DrfpEncoder.encode(df.rxn_smiles, mapping=True)
        pickle.dump(df, open(cache_file, "wb+"))
        pickle.dump(mapping, open(map_file, "wb+"))

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


def train_test_boost_model(
    df,
    split=0.1,
    variable="ec1",
) -> ConfusionMatrix:
    model_path = Path(Path(__file__).resolve().parent, "../models")
    model_file = Path(model_path, f"model-{variable}-{split}.keras")
    aux_file = Path(model_path, f"model-{variable}-{split}.pkl.gz")
    df_train, df_test = train_test_split(df, test_size=split)

    le = LabelEncoder()
    le.fit(np.concatenate([df_train[variable], df_test[variable]]))
    n_classes = len(le.classes_)

    X_train = np.array([x.astype(np.float32) for x in df_train.fps])
    X_test = np.array([x.astype(np.float32) for x in df_test.fps])

    y_train = le.transform(df_train[variable])

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = ConfusionMatrix(
        actual_vector=le.transform(df_test[variable]), predict_vector=y_pred
    )

    # model.save_model(str(model_file), num_iteration=model.best_iteration)

    pickle.dump(
        {"label_encoder": le, "confusion_matrix": cm},
        gzip.open(aux_file, "wb+"),
    )

    # cm.print_normalized_matrix()
    print(cm.overall_stat)
    print(f"Accuracy : {cm.overall_stat['Overall ACC']:.3f}")
    print(f"MCC : {cm.overall_stat['Overall MCC']:.3f}")
    print(f"CEN : {cm.overall_stat['Overall CEN']:.3f}")
    return cm


def train_test_model(
    df,
    split=0.1,
    variable="ec1",
    neurons: int = 1664,
    epochs: int = 10,
) -> ConfusionMatrix:
    model_path = Path(Path(__file__).resolve().parent, "../models")
    model_file = Path(model_path, f"model-{variable}-{split}.keras")
    aux_file = Path(model_path, f"model-{variable}-{split}.pkl.gz")
    df_train, df_test = train_test_split(df, test_size=split)

    le = LabelEncoder()
    le.fit(np.concatenate([df_train[variable], df_test[variable]]))
    n_classes = len(le.classes_)

    y_pred, model = get_mlp_model(
        np.array([x.astype(np.float32) for x in df_train.fps]),
        le.transform(df_train[variable]),
        np.array([x.astype(np.float32) for x in df_test.fps]),
        n_classes,
        neurons,
        epochs,
    )

    cm = ConfusionMatrix(
        actual_vector=le.transform(df_test[variable]), predict_vector=y_pred
    )

    model.save(str(model_file))

    pickle.dump(
        {"label_encoder": le, "confusion_matrix": cm},
        gzip.open(aux_file, "wb+"),
    )

    # cm.print_normalized_matrix()
    print(cm.overall_stat)
    print(f"Accuracy : {cm.overall_stat['Overall ACC']:.3f}")
    print(f"MCC : {cm.overall_stat['Overall MCC']:.3f}")
    print(f"CEN : {cm.overall_stat['Overall CEN']:.3f}")
    return cm


def train_test_multimodal_model(df, split=0.1, variable="ec1") -> ConfusionMatrix:
    model_path = Path(Path(__file__).resolve().parent, "../models")
    model_file = Path(model_path, f"model-{variable}-{split}.keras")
    aux_file = Path(model_path, f"model-{variable}-{split}.pkl.gz")
    df_train, df_test = train_test_split(df, test_size=split)

    le = LabelEncoder()
    le.fit(np.concatenate([df_train[variable], df_test[variable]]))
    n_classes = len(le.classes_)

    y_pred, model = get_multimodal_mlp_model(
        [
            np.array([x.astype(np.float32) for x in df_train.fps]),
            np.array([x.astype(np.float32) for x in df_train.dmqns]),
        ],
        le.transform(df_train[variable]),
        [
            np.array([x.astype(np.float32) for x in df_test.fps]),
            np.array([x.astype(np.float32) for x in df_test.dmqns]),
        ],
        n_classes,
    )

    cm = ConfusionMatrix(
        actual_vector=le.transform(df_test[variable]), predict_vector=y_pred
    )

    model.save(str(model_file))

    pickle.dump(
        {"label_encoder": le, "confusion_matrix": cm},
        gzip.open(aux_file, "wb+"),
    )

    # cm.print_normalized_matrix()
    print(cm.overall_stat)
    print(f"Accuracy : {cm.overall_stat['Overall ACC']:.3f}")
    print(f"MCC : {cm.overall_stat['Overall MCC']:.3f}")
    print(f"CEN : {cm.overall_stat['Overall CEN']:.3f}")
    return cm


@click.command()
@click.option("--variable", default="ec1", type=str)
@click.option("--split", default=0.1, type=float)
def main(variable, split):
    df = load_data()
    df = calculate_cache_drfp(df)
    df = calculate_cache_mqn(df)
    df = calculate_cache_rxnfp(df)

    # concatenate the two fps
    # fps = []
    # for drfp, mqn in zip(df.fps, df.dmqns):
    #     fps.append(np.concatenate((drfp, mqn)))

    # df["fps"] = df.dmqns
    # cm = train_test_model(df, variable=variable, split=split)
    # cm = train_test_multimodal_model(df, variable=variable, split=split)
    cm = train_test_boost_model(df, variable=variable, split=split)

    cm_path = Path(Path(__file__).resolve().parent, "../models/mlp.cm")
    with open(cm_path, "wb") as f:
        pickle.dump(cm, f)


if __name__ == "__main__":
    main()
