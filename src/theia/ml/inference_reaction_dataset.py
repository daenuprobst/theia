from typing import List

from torch import FloatTensor
from torch.utils.data import Dataset

from drfp import DrfpEncoder

import numpy as np


class InferenceReactionDataset(Dataset):
    def __init__(
            self, rxns: List, 
            label: str = "label", 
            root_central_atom: bool=False, 
            radius: int = 2, 
            include_hydrogens: bool = True, 
            n_folded_length: int = 10240
    ):
        self.rxns = rxns
        self.size = len(rxns)
        self.label = label

        fps, shinglings, mappings = DrfpEncoder.encode(
            rxns,
            mapping=True,
            atom_index_mapping=True,
            root_central_atom=root_central_atom,
            radius=radius,
            include_hydrogens=include_hydrogens,
            n_folded_length=n_folded_length,
        )

        self.mappings = mappings
        self.shinglings = shinglings
        self.X = FloatTensor(
            np.array([x.astype(np.float32) for x in fps], dtype=np.float32)
        )

    def __getitem__(self, i):
        return self.X[i]

    def __len__(self):
        return self.size
