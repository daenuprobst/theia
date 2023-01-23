import re
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator,
    get_default_model_and_tokenizer,
    generate_fingerprints,
)


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


def main():
    df = load_data()
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

    cache_path = Path(__file__).resolve().parent
    cache_file = Path(cache_path, "../data/ecreact-rxnfp.cache.pkl")
    if cache_file.exists():
        df = pickle.load(open(cache_file, "rb"))
    else:
        rxnfps = []
        for rxn_smiles in tqdm(df.rxn_smiles):
            rxnfps.append(np.array(rxnfp_generator.convert(rxn_smiles)))

        df["rxnfps"] = rxnfps
        pickle.dump(df, open(cache_file, "wb+"))

    return df


if __name__ == "__main__":
    main()
