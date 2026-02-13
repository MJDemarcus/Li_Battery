import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class BatteryLSTM:
    def __init__(self, input_shape, units=50, dropout_rate=0.2, demo_mode=False):
        self.model = Sequential()
        
        if demo_mode:
            # Lightweight architecture for fast demo training
            self.model.add(LSTM(units=units, return_sequences=False, input_shape=input_shape))
            self.model.add(Dropout(dropout_rate))
        else:
            # Deep architecture for production/research
            self.model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
            self.model.add(Dropout(dropout_rate))
            self.model.add(LSTM(units=units, return_sequences=True))
            self.model.add(Dropout(dropout_rate))
            self.model.add(LSTM(units=units, return_sequences=False))
            self.model.add(Dropout(dropout_rate))
        
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None, verbose=0, **kwargs):
        """
        Trains the model.
        """
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=verbose, **kwargs)
    
    def predict(self, X):
        """
        Generates predictions.
        """
        return self.model.predict(X)
    
    def save(self, path):
        """
        Saves the model to disk.
        """
        self.model.save(path)
        
    @staticmethod
    def load(path):
        """
        Loads a model from disk.
        """
        return tf.keras.models.load_model(path)
class MockBatteryModel:
    """
    A mock model for instant demo feedback without TensorFlow overhead.
    """
    def __init__(self, input_shape, units=50, dropout_rate=0.2, demo_mode=True):
        self.input_shape = input_shape
        self.history = {'loss': [0.1, 0.05, 0.02, 0.01] * 3 } # Fake history
        self.model = self # Mock compatibility with code expecting model.model

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None, verbose=0, **kwargs):
        # Simulate training time slightly
        import time
        time.sleep(0.5)
        
        # Return a fake history object structure
        class History:
            def __init__(self):
                self.history = {'loss': [0.5 * (0.8 ** i) for i in range(epochs)]} # Decay curve
        return History()
    
    def predict(self, X):
        # Return input values with some processing to look like prediction
        # For auto-encoder style (reconstruction), we just return input + noise
        # But X is (batch, seq, features), we need (batch, 1) or similar.
        # Assuming we predict the NEXT step, or if it's anomaly detection (reconstruction).
        # Let's assume prediction of next step.
        
        # Simple heuristic: average of last step + small noise
        last_step = X[:, -1, 0] # Take first feature of last step
        noise = np.random.normal(0, 0.01, size=last_step.shape)
        return (last_step + noise).reshape(-1, 1)

    def save(self, path):
        pass
        
    @staticmethod
    def load(path):
        return MockBatteryModel(None)
