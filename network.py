from tensorflow import keras
import numpy as np

class Network:
    def __init__(self, layer_sizes):
        """
        Inicializa a rede neural com as camadas especificadas
        layer_sizes: lista com o número de neurônios em cada camada [entrada, ...ocultas, saída]
        """
        self.layer_sizes = layer_sizes
        self._build_model()
        
    def _build_model(self):
        """Constrói o modelo usando Keras"""
        self.model = keras.Sequential()
        
        # Adiciona a camada de entrada e primeira camada oculta
        self.model.add(keras.layers.Dense(
            self.layer_sizes[1],
            activation='relu',
            input_shape=(self.layer_sizes[0],)
        ))
        
        # Adiciona as camadas ocultas restantes
        for size in self.layer_sizes[2:-1]:
            self.model.add(keras.layers.Dense(size, activation='relu'))
            
        # Adiciona a camada de saída
        self.model.add(keras.layers.Dense(self.layer_sizes[-1], activation='sigmoid'))
        
        # Compila o modelo
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, validation_data=None, epochs=50, batch_size=32):
        """Treina a rede neural"""
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
    
    def predict(self, X):
        """Faz previsões usando a rede neural"""
        return self.model.predict(X)

    def get_weights(self):
        """Retorna os pesos da rede"""
        return [layer.get_weights() for layer in self.model.layers]

    def set_weights(self, weights):
        """Define os pesos da rede"""
        for layer, w in zip(self.model.layers, weights):
            layer.set_weights(w)
            
    def get_layer_output(self, X, layer_index):
        """Retorna a saída de uma camada específica"""
        intermediate_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[layer_index].output
        )
        return intermediate_model.predict(X)