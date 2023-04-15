from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QLineEdit, QLabel, QVBoxLayout, QHBoxLayout
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from qt_material import apply_stylesheet
import qtawesome as qta


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Ejemplo de pantalla dividida en tres columnas')
        
        # Configurar los estilos de los botones
        apply_stylesheet(self, theme='light_pink.xml')

        # Configurar las dimensiones y la posición de la ventana
        self.showMaximized()
        
        # Agregar una cámara web en la tercera sección de la pantalla
        self.video_label = QLabel(self)
        self.video_label.resize(2000, 2000)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Agregar tres botones en la primera sección de la pantalla
        button1 = QPushButton(qta.icon('fa5s.copy', options=[{'scale_factor': 1.2}]),'')
        button2 = QPushButton(qta.icon('fa5s.trash', options=[{'scale_factor': 1.2}]),'')
        button3 = QPushButton(qta.icon('fa5s.music', options=[{'scale_factor': 1.2}]),'')
        
        button1.setFixedSize(100,100)
        button2.setFixedSize(100,100)
        button3.setFixedSize(100,100)
        
        button1.setStyleSheet("font-size: 20px")
        button2.setStyleSheet("font-size: 20px")
        button3.setStyleSheet("font-size: 20px")

        # Agregar un cuadro de texto no editable en la segunda sección de la pantalla
        text_edit = QLineEdit()
        text_edit.setPlaceholderText('Aquí aparecerá su mensaje')
        text_edit.setFixedSize(700,800)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-size: 20px")
        text_edit.setAlignment(Qt.AlignTop)

        # Agregar un label debajo de la cámara web con el título "Letra detectada"
        title_label = QLabel('Letra detectada')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 32px; font: bold")

        # Crear los layouts para organizar los elementos de la interfaz de usuario
        boton_layout = QVBoxLayout()
        boton_layout.addWidget(button1)
        boton_layout.addWidget(button2)
        boton_layout.addWidget(button3)
        boton_layout.setAlignment(Qt.AlignCenter)

        mensaje_layout = QVBoxLayout()
        mensaje_layout.addWidget(text_edit)
        mensaje_layout.setAlignment(Qt.AlignCenter)

        camara_layout = QVBoxLayout()
        camara_layout.addWidget(self.video_label)
        camara_layout.addWidget(title_label)
        camara_layout.setAlignment(Qt.AlignCenter)

        # Crear un layout QGridLayout para la interfaz de usuario
        principal_layout = QGridLayout()
        principal_layout.setSpacing(20)
        principal_layout.addLayout(boton_layout, 0, 0, Qt.AlignCenter)
        principal_layout.addLayout(mensaje_layout, 0, 1, Qt.AlignCenter)
        principal_layout.addLayout(camara_layout, 0, 2, Qt.AlignCenter)

        # Establecer el layout principal de la ventana
        self.setLayout(principal_layout)

        self.show()

        # Iniciar la captura de video desde la cámara web
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.actualizar_imagen)
        self.timer.start(30)

    def actualizar_imagen(self):
        """Actualizar la imagen de la cámara web en el label correspondiente."""
        ret, frame = self.capture.read()
        frame = cv2.resize(frame, (800, 800))
        if ret:
            imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen = cv2.flip(imagen, 1)
            h, w, c = imagen.shape
            
            # Convertir la imagen en un QPixmap para poder mostrarla en un QLabel
            qimg = QImage(imagen.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Mostrar la imagen en el QLabel correspondiente
            self.video_label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication([])
    window = MyApp()
    app.exec_()