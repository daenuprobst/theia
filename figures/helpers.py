import string
from matplotlib.axes import Axes
from matplotlib import rcParams


def index_subplots(
    axs: Axes,
    font_size: float = 20,
    font_weight: str = "bold",
    font_family: str = rcParams["font.family"],
    x: float = -0.1,
    y: float = 1.1,
    uppercase: bool = False,
    prefix: str = "",
    suffix: str = "",
    offset: int = 0,
):
    int_to_char_map = string.ascii_lowercase
    if uppercase:
        int_to_char_map = string.ascii_uppercase

    for i, ax in enumerate(axs):
        ax.text(
            x,
            y,
            prefix + int_to_char_map[i + offset] + suffix,
            transform=ax.transAxes,
            size=font_size,
            weight=font_weight,
            fontfamily=font_family,
        )
