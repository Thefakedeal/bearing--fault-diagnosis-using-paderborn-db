import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from src.models.base_model import BaseModel


class InceptionTimeModel(BaseModel):

    def __init__(self, num_classes, input_shape, lr=1e-3):
        """
        num_classes: number of output classes
        input_shape: (seq_len, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.lr = lr
        self.model = self._build_model()

    # --------------------------------------------------
    # Inception block
    # --------------------------------------------------
    def _inception_module(self, x, filters=32):

        conv1 = layers.Conv1D(filters, kernel_size=10, padding='same', activation='relu')(x)
        conv2 = layers.Conv1D(filters, kernel_size=20, padding='same', activation='relu')(x)
        conv3 = layers.Conv1D(filters, kernel_size=40, padding='same', activation='relu')(x)

        pool = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
        pool = layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')(pool)

        x = layers.Concatenate()([conv1, conv2, conv3, pool])
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        return x

    # --------------------------------------------------
    # Build full model
    # --------------------------------------------------
    def _build_model(self):

        inputs = layers.Input(shape=self.input_shape)

        x = self._inception_module(inputs)
        x = self._inception_module(x)
        x = self._inception_module(x)

        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    # --------------------------------------------------
    # Required functions
    # --------------------------------------------------
    def train(self, X, y, epochs=20, batch_size=32, validation_data=None):
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )

    def predict(self, X):
        probs = self.model.predict(X)
        return np.argmax(probs, axis=1)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def export(self, path):
        # export only feature extractor (without final softmax layer)
        feature_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-2].output
        )
        feature_model.save(path)
