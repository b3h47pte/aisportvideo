import logging

import typer

from aisportvideo.cli.detect import app as detect_app
from aisportvideo.cli.models import app as models_app
from aisportvideo.cli.pose import app as pose_app

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
)

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(detect_app, name="detect")
app.add_typer(pose_app, name="pose")
app.add_typer(models_app, name="models")


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        main()


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
