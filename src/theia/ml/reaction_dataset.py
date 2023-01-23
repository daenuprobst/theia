import numpy as np
import pandas as pd

from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset


class ReactionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label: str = "label"):
        self.size = len(df)
        self.label = label
        self.X = FloatTensor(
            np.array([x.astype(np.float32) for x in df.fps], dtype=np.float32)
        )
        self.y = LongTensor(df[self.label].to_numpy(dtype=np.int32))
        self.fps = df["fps"]
        self.rxn_smiles = df["rxn_smiles"]

    def __getitem__(self, i):
        return (
            self.X[i],
            self.y[i],
        )

    def __len__(self):
        return self.size
