from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSpinBox, QLabel, QListWidget
)
from PyQt6.QtCore import Qt

class NetworkConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuração da Rede Neural")
        
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Lista de camadas
        self.layers_list = QListWidget()
        self.layers_list.addItem("Camada de entrada: 13 neurônios (fixo)")
        self.layers_list.addItem("Camada de saída: 1 neurônio (fixo)")
        layout.addWidget(QLabel("Arquitetura atual:"))
        layout.addWidget(self.layers_list)
        
        # Controles para adicionar camada
        controls = QHBoxLayout()
        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(1, 100)
        self.neurons_spin.setValue(5)
        controls.addWidget(QLabel("Número de neurônios:"))
        controls.addWidget(self.neurons_spin)
        
        # Botões
        btn_add = QPushButton("Adicionar Camada Oculta")
        btn_add.clicked.connect(self.add_hidden_layer)
        controls.addWidget(btn_add)
        
        btn_remove = QPushButton("Remover Última Camada Oculta")
        btn_remove.clicked.connect(self.remove_hidden_layer)
        controls.addWidget(btn_remove)
        
        layout.addLayout(controls)
        
        # Botão de confirmação
        btn_confirm = QPushButton("Confirmar e Iniciar")
        btn_confirm.clicked.connect(self.confirm_architecture)
        layout.addWidget(btn_confirm)
        
        self.hidden_layers = []
        self.resize(600, 400)
    
    def add_hidden_layer(self):
        neurons = self.neurons_spin.value()
        # Inserir antes da camada de saída
        pos = self.layers_list.count() - 1
        self.layers_list.insertItem(pos, f"Camada oculta: {neurons} neurônios")
        self.hidden_layers.append(neurons)
    
    def remove_hidden_layer(self):
        if len(self.hidden_layers) > 0:
            # Remover antes da camada de saída
            pos = self.layers_list.count() - 2
            self.layers_list.takeItem(pos)
            self.hidden_layers.pop()
    
    def confirm_architecture(self):
        # Retorna a arquitetura completa [13, ...hidden_layers, 1]
        architecture = [13] + self.hidden_layers + [1]
        self.architecture = architecture
        self.close()

def get_network_architecture():
    """Função para abrir a janela de configuração e retornar a arquitetura escolhida"""
    app = QApplication([])
    window = NetworkConfigWindow()
    window.show()
    app.exec()
    return getattr(window, 'architecture', [13, 1])  # padrão se usuário fechar sem confirmar