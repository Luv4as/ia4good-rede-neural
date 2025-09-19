from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
    QVBoxLayout, QWidget, QGraphicsEllipseItem, QGraphicsLineItem
)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter
import sys
import numpy as np

class NeuronItem(QGraphicsEllipseItem):
    def __init__(self, x, y, size=30):
        super().__init__(-size/2, -size/2, size, size)
        self.setPos(x, y)
        self.setBrush(QBrush(QColor('#3498db')))
        self.setPen(QPen(Qt.PenStyle.NoPen))

class NetworkScene(QGraphicsScene):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.draw_network()

    def draw_network(self):
        # Dimensões e espaçamento
        neuron_size = 30
        layer_spacing = 200
        neuron_spacing = 60
        
        # Para cada camada da rede
        for i, layer in enumerate(self.model.layers):
            n_neurons = len(layer.neurons) if hasattr(layer, 'neurons') else layer.units
            layer_height = (n_neurons - 1) * neuron_spacing
            
            # Para cada neurônio na camada
            for j in range(n_neurons):
                # Calcular posição do neurônio
                x = i * layer_spacing + 100
                y = -layer_height/2 + j * neuron_spacing + 100
                
                # Criar e adicionar neurônio
                neuron = NeuronItem(x, y)
                self.addItem(neuron)
                
                # Adicionar conexões com a camada anterior
                if i > 0:
                    prev_layer = self.model.layers[i-1]
                    n_prev_neurons = len(prev_layer.neurons) if hasattr(prev_layer, 'neurons') else prev_layer.units
                    prev_height = (n_prev_neurons - 1) * neuron_spacing
                    
                    for k in range(n_prev_neurons):
                        prev_x = (i-1) * layer_spacing + 100
                        prev_y = -prev_height/2 + k * neuron_spacing + 100
                        
                        # Desenhar linha de conexão
                        line = QGraphicsLineItem(prev_x + neuron_size/2, prev_y,
                                              x - neuron_size/2, y)
                        line.setPen(QPen(QColor('#bdc3c7')))
                        self.addItem(line)

class NetworkVisualizer(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Visualizador de Rede Neural')
        self.setGeometry(100, 100, 800, 600)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Cena e visualização
        scene = NetworkScene(self.model)
        view = QGraphicsView(scene)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(view)

def show_network(model):
    app = QApplication(sys.argv)
    viewer = NetworkVisualizer(model)
    viewer.show()
    app.exec()