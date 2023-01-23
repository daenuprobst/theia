import copy
import math
from collections import defaultdict

try:
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    cm = None
except RuntimeError:
    cm = None

import numpy

from rdkit import Chem
from rdkit import DataStructs
from rdkit import Geometry
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw.MolDrawing import DrawingOptions


def ExplainMolecule(
    mol,
    weights,
    draw2d,
    colorMap=None,
    sigma=None,
    contourLines=10,
):
    """
    Generates the similarity map for a molecule given the atomic weights.
    Parameters:
      mol -- the molecule of interest
      colorMap -- the matplotlib color map scheme, default is custom PiWG color map
      scale -- the scaling: scale < 0 -> the absolute maximum weight is used as maximum scale
                            scale = double -> this is the maximum scale
      size -- the size of the figure
      sigma -- the sigma for the Gaussians
      coordScale -- scaling factor for the coordinates
      step -- the step for calcAtomGaussian
      colors -- color of the contour lines
      contourLines -- if integer number N: N contour lines are drawn
                      if list(numbers): contour lines at these numbers are drawn
      alpha -- the alpha blending value for the contour lines
      kwargs -- additional arguments for drawing
    """
    # for atom in mol.GetAtoms():
    #     atom.SetAtomMapNum(atom.GetIdx())

    # if mol.GetNumAtoms() < 2:
    #     raise ValueError("too few atoms")

    mol = rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)

    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)

    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = (
                0.3
                * (
                    mol.GetConformer().GetAtomPosition(idx1)
                    - mol.GetConformer().GetAtomPosition(idx2)
                ).Length()
            )
        else:
            if mol.GetNumHeavyAtoms() == 1:
                sigma = 0.3
            else:
                sigma = (
                    0.3
                    * (
                        mol.GetConformer().GetAtomPosition(0)
                        - mol.GetConformer().GetAtomPosition(1)
                    ).Length()
                )
        sigma = round(sigma, 2)

    sigmas = [sigma] * mol.GetNumAtoms()
    locs = []

    for i in range(mol.GetNumAtoms()):
        p = mol.GetConformer().GetAtomPosition(i)
        locs.append(Geometry.Point2D(p.x, p.y))

    draw2d.ClearDrawing()

    ps = Draw.ContourParams()
    ps.setScale = True
    ps.fillGrid = True
    ps.gridResolution = 0.1
    ps.extraGridPadding = 0.5

    if colorMap is not None:
        if cm is not None and isinstance(colorMap, type(cm.Blues)):
            # it's a matplotlib colormap:
            clrs = [tuple(x) for x in colorMap([0, 0.5, 1])]
        elif type(colorMap) == str:
            if cm is None:
                raise ValueError(
                    "cannot provide named colormaps unless matplotlib is installed"
                )
            clrs = [tuple(x) for x in cm.get_cmap(colorMap)([0, 0.5, 1])]
        else:
            clrs = [colorMap[0], colorMap[1], colorMap[2]]
        ps.setColourMap(clrs)

    # The molecule has to be drawn first to set the size of the canvas,
    # Draw.ContourAndDrawGaussians will overwrite it though, so after that,
    # it is drawn again
    draw2d.drawOptions().useBWAtomPalette()
    draw2d.drawOptions().clearBackground = False
    # draw2d.drawOptions().scalingFactor = 50.0
    # draw2d.drawOptions().padding = 5.0 # Setting this makes it run forever, weird.
    # draw2d.DrawMolecule(mol)
    draw2d.ClearDrawing()

    Draw.ContourAndDrawGaussians(
        draw2d, locs, weights, sigmas, nContours=contourLines, params=ps, mol=mol
    )

    draw2d.DrawMolecule(mol)
    draw2d.FinishDrawing()

    return draw2d
