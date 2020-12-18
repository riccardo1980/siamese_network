import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def pair_plot(
    pairs: List[np.array],
    labels: List[np.array],
    items_per_row: int = 3,
    label_map: Tuple[str] = ('different', 'same')
):
    """
        Plot pairs and labels

        :param pairs: np.array containing images pairs
        :param labels: np.array of labels
        :param items_per_row: number of items per row in visualization
        :param label_map: lsit of string labels

    """
    N = len(pairs)
    rows = (N + items_per_row - 1) // items_per_row
    layout = (rows, items_per_row)
    fig, axs = plt.subplots(*layout, squeeze=False)

    for idx, (pair, lbl) in enumerate(zip(pairs, labels)):
        row = idx // items_per_row
        col = idx % items_per_row

        img = np.hstack(pair)
        axs[row, col].imshow(img)
        axs[row, col].axis('off')
        axs[row, col].set_title(label_map[lbl[0]])
