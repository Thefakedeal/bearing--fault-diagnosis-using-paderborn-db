from src.models.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers, models

class Simple1DCNNModel(BaseModel):
    def __init__(self, num_classes, input_shape, lr=1e-3):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.lr = lr
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv1D(32, 3, activation='relu')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def train(self, X, y, epochs=20, batch_size=32):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stop]
        )

        return history

    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        self.model.save(path)
        
    def load(self, path):
        self.model = tf.keras.models.load_model(path)