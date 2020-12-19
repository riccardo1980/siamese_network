from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D
)


def build_siamese_model(
    input_shape: Tuple[int],
    embedding_dim: int = 48,
    dropout_rate: float = 0.3
) -> Model:
    """
        Builds a feature extractor

        :params input_shape: shape of input (H W C)
        :params embedding_dim: shape of output (C)
        :params_dorpout rate: dropout probability

    """
    inputs = Input(input_shape)

    x = Conv2D(64, (2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    pooled_output = GlobalAveragePooling2D()(x)

    outputs = Dense(embedding_dim)(pooled_output)

    model = Model(inputs, outputs)

    return model
