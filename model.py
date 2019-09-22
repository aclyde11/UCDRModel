import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

def get_model(vocab_size, emb_size, max_len, h_size, dropout=0.2):
    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, emb_size, input_length=max_len)(inputs)
    x = layers.SimpleRNN(h_size, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
                           recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                           kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                           recurrent_dropout=0.0, return_sequences=True, return_state=True, go_backwards=False,
                           stateful=False, unroll=False)(x)
    x = layers.SimpleRNN(h_size, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
                           recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                           kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                           recurrent_dropout=0.0, return_sequences=True, return_state=True, go_backwards=False,
                           stateful=False, unroll=False)(x)
    x = layers.SimpleRNN(h_size, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
                           recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                           kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                           recurrent_dropout=0.0, return_sequences=True, return_state=True, go_backwards=False,
                           stateful=False, unroll=False)(x)
    x = layers.SimpleRNN(h_size, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
                           recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                           kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0,
                           recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False,
                           stateful=False, unroll=False)(x)


    x = layers.Dense(h_size, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    predictions = layers.Dense(1, activation=None)(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    return model
