from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


@app.command(name="object")
def detect_objects(path: Annotated[Path, typer.Argument(help="Path to the video file")]):
    pass
