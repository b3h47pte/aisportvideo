
from PySide6 import QtGui, QtWidgets

def center_widget_on_screen(widget: QtWidgets.QWidget):
    centerPoint = QtGui.QScreen.availableGeometry(QtWidgets.QApplication.primaryScreen()).center()
    fg = widget.frameGeometry()
    fg.moveCenter(centerPoint)
    widget.move(fg.topLeft())