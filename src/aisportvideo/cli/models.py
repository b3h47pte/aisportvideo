import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer
from ultralytics import YOLO

from aisportvideo.models.yolo import (
    HF_REPOSITORY,
    YoloModelSize,
    YoloModelType,
    exported_model_type_to_yolo_format,
    get_yolo_hf_filename,
)
from aisportvideo.utils.models import extension_to_model_type

app = typer.Typer()

_logger = logging.getLogger("aisportvideo.cli.models")


@app.command(name="yolo11")
def prepare_yolo11(
    size: Annotated[YoloModelSize, typer.Option(help="Size of the model")],
    model_type: Annotated[YoloModelType, typer.Option(help="Type of the model")],
    output_path: Annotated[Path, typer.Option(help="Path to save the model")],
):
    from huggingface_hub import hf_hub_download

    raw_model_path = Path(
        hf_hub_download(
            HF_REPOSITORY, filename=get_yolo_hf_filename(size=size, model_type=model_type)
        )
    )

    assert raw_model_path.is_file()
    _logger.info("Model downloaded to %s", raw_model_path)

    exported_model_type = extension_to_model_type(output_path.suffix)

    model = YOLO(raw_model_path)
    model.export(format=exported_model_type_to_yolo_format(model_type=exported_model_type))

    exported_model_path = raw_model_path.with_suffix(output_path.suffix)

    if not exported_model_path.is_file():
        err = f"Model not exported to {exported_model_path}"
        raise RuntimeError(err)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported_model_path, output_path)
    if not output_path.is_file():
        err = f"Failed to move model to {output_path}"
        raise RuntimeError(err)
