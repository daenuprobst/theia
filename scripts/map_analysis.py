import pickle
import multiprocessing as mp
from collections import defaultdict
from functools import partial
import typer
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from rdkit.Chem import rdChemReactions
from rdkit.Chem import AllChem


def load_reactions(path: str):
    df = pd.read_csv(path)

    # Replace ecreact style reactions with standard reaction SMILES
    df["rxn"] = df.rxn.str.replace(r"\|.*>", ">>", regex=True)
    df = df[df.ec.notna()]

    df[["ec_1", "ec_2", "ec_3", "ec_4"]] = df.ec.str.split(".", expand=True)
    df = df[df.ec_1 != "7"]
    return df


def reactions_to_mols(df: pd.DataFrame):
    mols = []

    for smiles in df.rxn:
        rdrxn = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
        rxnmols = []
        for mol in rdrxn.GetReactants():
            rxnmols.append(mol)
        for mol in rdrxn.GetProducts():
            rxnmols.append(mol)
        mols.append(rxnmols)

    return mols


def get_count(mols, m):
    count = 0
    subs = defaultdict(set)

    substruct_cache = {}

    # for _, mols in enumerate(rxn_mols):
    #     subs = defaultdict(set)
    for mol in mols:
        for key, value in m.items():
            if len(value) <= 1:
                continue
            for substruct in value:
                if substruct in substruct_cache:
                    s = substruct_cache[substruct]
                else:
                    s = AllChem.MolFromSmarts(substruct)

                if s:
                    if mol.HasSubstructMatch(s):
                        subs[key].add(substruct)

    for key, value in subs.items():
        if len(value) > 1:
            count += 1

    return count


def main(map_file: str, reactions_file: str):
    m = pickle.load(open(map_file, "rb"))
    df = load_reactions(reactions_file)

    rxn_mols = reactions_to_mols(df)

    counts = process_map(partial(get_count, m=m), rxn_mols, chunksize=25)

    print(sum(counts))


if __name__ == "__main__":
    typer.run(main)
