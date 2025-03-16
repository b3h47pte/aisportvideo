import random

from PySide6 import QtCore, QtWidgets


class HomeWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QtWidgets.QPushButton("Click me!")
        self.text = QtWidgets.QLabel("Hello World", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text)
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.button.clicked.connect(self.magic)

    @QtCore.Slot()
    def magic(self):
        self.text.setText(random.choice(self.hello))
