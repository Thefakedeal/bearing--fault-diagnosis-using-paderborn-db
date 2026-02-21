import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn1d(input_shape = (2048, 1)):

    model = models.Sequential()
    model.add(layers.Conv1D(
        filters = 32,
        kernel_size = 7,
        strides = 1,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size = 2))

    model.add(layers.Conv1D(
        filters = 64,
        kernel_size = 5,
        strides = 1,
        padding = 'same',
        activation = 'relu',
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size = 2))

    model.add(layers.Conv1D(
        filters = 128,
        kernel_size = 3,
        strides = 1,
        padding = 'same',
        activation = 'relu',
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size = 2))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
        loss = 'binary_crossentropy',
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ]
    )
    return model
