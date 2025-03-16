import typer

app = typer.Typer()


@app.command()
def main():
    import sys

    from PySide6 import QtWidgets

    from aisportvideo.ui.home import HomeWidget
    from aisportvideo.utils.layout import center_widget_on_screen

    qtapp = QtWidgets.QApplication([])

    widget = HomeWidget()
    widget.resize(1600, 900)
    widget.show()
    center_widget_on_screen(widget)
    sys.exit(qtapp.exec())


if __name__ == "__main__":
    app()
