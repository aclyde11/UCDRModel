from keras.layers import Input, Dense, Dropout
from keras.models import Model

def get_model(input_size=784, dropout=0.2):
    inputs = Input(shape=(input_size,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(512, activation='relu')(inputs)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout * 0.8)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout * 0.7)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout * 0.6)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout * 0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout * 0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout * 0.2)(x)

    predictions = Dense(1, activation=None)(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    return model
