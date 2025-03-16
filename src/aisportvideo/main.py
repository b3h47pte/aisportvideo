import typer

app = typer.Typer()


@app.command()
def main():
    import sys

    from PySide6 import QtGui, QtWidgets

    from aisportvideo.ui.home import HomeWidget

    qtapp = QtWidgets.QApplication([])

    widget = HomeWidget()
    widget.resize(1600, 900)

    centerPoint = QtGui.QScreen.availableGeometry(QtWidgets.QApplication.primaryScreen()).center()
    fg = widget.frameGeometry()
    fg.moveCenter(centerPoint)
    widget.move(fg.topLeft())

    widget.show()
    sys.exit(qtapp.exec())


if __name__ == "__main__":
    app()
