def train_mlp_model(model, x_train, y_train, batch_size=16, epochs=30, validation_split=0.2, optimizer='adam',loss='mse'):
    """
    Train multilayer perceptron model on given data and return trained model
    """
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    return model
