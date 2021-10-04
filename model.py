import math
import tensorflow as tf

def cnn_model_builder():
    tf.keras.backend.clear_session()

    MAX_LEN = 60
    VOCAB_SIZE = 20
    HIDDEN_UNITS = MAX_LEN//2

    kernel_size_ = 15
    filters_ = 30
    pool_size_ = 2

    inp = tf.keras.layers.Input(shape=(MAX_LEN, VOCAB_SIZE))
    l1 = tf.keras.layers.Conv1D(filters=filters_, kernel_size=kernel_size_, activation="relu", padding='same',)(inp)
    l3 = tf.keras.layers.MaxPooling1D(pool_size=pool_size_)(l1)
    l4 = tf.keras.layers.Conv1D(filters=filters_, kernel_size=kernel_size_, activation="relu", padding='same',)(l3)
    l5 = tf.keras.layers.MaxPooling1D(pool_size=pool_size_)(l4)
    l6 = tf.keras.layers.Flatten()(l5)
    l7 = tf.keras.layers.Dense(HIDDEN_UNITS, activation='linear')(l6)
    #decoder
    d1 = tf.keras.layers.Dense(MAX_LEN//2//2*filters_, activation='relu')(l7)
    d2 = tf.keras.layers.Reshape((MAX_LEN//2//2, filters_))(d1)
    d3 = tf.keras.layers.UpSampling1D(size=pool_size_)(d2)
    d4 = tf.keras.layers.Conv1D(filters=filters_, kernel_size=kernel_size_, activation="relu", padding='same')(d3)
    d5 = tf.keras.layers.UpSampling1D(size=pool_size_)(d4)
    decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE, activation='softmax'))(d5)

    model = tf.keras.models.Model(inp, decoded)
    model.load_weights('protein_autoencoder.epoch04-loss1.80.hdf5')
    return model

def lstm_model_builder():
    tf.keras.backend.clear_session()
    MAX_LEN = 60
    VOCAB_SIZE = 20

    inputs  = tf.keras.layers.Input(shape=(MAX_LEN, VOCAB_SIZE))
    encoded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(inputs)
    encoded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(encoded)

    decoded = tf.keras.layers.RepeatVector(MAX_LEN)(encoded)
    decoded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(decoded)
    decoded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(decoded)
    decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(VOCAB_SIZE, activation = 'softmax'))(decoded)

    model = tf.keras.models.Model(inputs, decoded)
    model.load_weights('lstm_protein_autoencoder.epoch04-loss0.08.hdf5')
    return model
