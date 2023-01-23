from typing import Tuple, List, Dict
from dataclasses import dataclass
from rdkit.Chem import AllChem

import numpy as np


@dataclass
class ExplainedReaction:
    rxn_smiles: str
    label: str
    raw_weights: np.ndarray
    reactant_weights: List[List[float]]
    product_weights: List[List[float]]
    neg_reactant_weights: List[List[float]]
    neg_product_weights: List[List[float]]
    pos_reactant_weights: List[List[float]]
    pos_product_weights: List[List[float]]
    reactant_fragments: List[Tuple[List[str], float]]
    product_fragments: List[Tuple[List[str], float]]
    top_missing_fragments: List[Tuple[List[str], float]]
    prediction_probability: float

    def remove_explicit_hydrogens(self):
        self.__remove_explicit_hydrogens(self.reactant_fragments)
        self.__remove_explicit_hydrogens(self.product_fragments)
        self.__remove_explicit_hydrogens(self.top_missing_fragments)

        return self

    def __remove_explicit_hydrogens(self, data):
        for i in range(len(data)):
            for j in range(len(data[i][0])):
                mol = AllChem.MolFromSmiles(data[i][0][j])
                if mol:
                    data[i][0][j] = AllChem.MolToSmiles(mol)

    def summed_weights(self) -> float:
        return np.sum(self.raw_weights)

    def abs_neg_pos(self) -> Tuple[float]:
        neg = 0.0
        pos = 0.0

        for val in self.raw_weights:
            if val < 0.0:
                neg += val
            else:
                pos += val

        return (neg, pos)

    def reactant_weights_string(self) -> str:
        return self.__weights_to_string(self.reactant_weights)

    def product_weights_string(self) -> str:
        return self.__weights_to_string(self.product_weights)

    def neg_reactant_weights_string(self) -> str:
        return self.__weights_to_string(self.neg_reactant_weights)

    def neg_product_weights_string(self) -> str:
        return self.__weights_to_string(self.neg_product_weights)

    def pos_reactant_weights_string(self) -> str:
        return self.__weights_to_string(self.pos_reactant_weights)

    def pos_product_weights_string(self) -> str:
        return self.__weights_to_string(self.pos_product_weights)

    def __weights_to_string(self, weights) -> str:
        return ";".join(
            [
                ",".join(
                    map(
                        str,
                        [weights[idx][i] for i in range(len(weights[idx]))],
                    )
                )
                for idx in range(len(weights))
            ]
        )
