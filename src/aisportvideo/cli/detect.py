from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


@app.command(name="object")
def detect_objects(path: Annotated[Path, typer.Argument(help="Path to the video file")]):
    from aisportvideo.models.yolo import YoloModelSize, YoloModelType, load_yolo_from_assets
    from aisportvideo.utils.models import ModelType

    model = load_yolo_from_assets(
        size=YoloModelSize.SMALL, model_type=YoloModelType.DETECTION, export_type=ModelType.ONNX
    )