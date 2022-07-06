import numpy as np


def train_mlp_model(model: object, x_train: np.array, y_train: np.array, batch_size: int = 16,
                    epochs: int = 30, validation_split: float = 0.2, optimizer: str = 'adam', loss: str = 'mse'):
    """
    Train multilayer perceptron model on given data and return trained model
    """

    model.compile(optimizer=optimizer, loss=loss)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    return model
