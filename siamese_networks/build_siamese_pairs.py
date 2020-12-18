import numpy as np
from typing import List, Tuple


def make_pairs(
    images: List[np.array],
    labels: List[np.array]
) -> Tuple[np.array, np.array]:
    """
        Generate list of pairs and relative labels

        :param images: List of np.array containing images
        :param labels: List of np.array labels

        :return: (features, labels)
    """

    pair_images = []
    pair_labels = []

    num_classes = len(np.unique(labels))

    # get list of indices, one entry for label
    idx_by_label = [np.where(labels == i)[0] for i in range(0, num_classes)]

    for A_idx in range(len(images)):
        # get A image and it's label
        A_image = images[A_idx]
        A_label = labels[A_idx]

        # positive example: needs to be extracted from same label images
        pos_idx = np.random.choice(idx_by_label[A_label])
        pos_image = images[pos_idx]

        pair_images.append([A_image, pos_image])
        pair_labels.append([1])

        # negative example
        idx_by_neg = np.where(labels != A_label)[0]
        neg_idx = np.random.choice(idx_by_neg)
        neg_image = images[neg_idx]

        pair_images.append([A_image, neg_image])
        pair_labels.append([0])

    return (np.array(pair_images), np.array(pair_labels))
