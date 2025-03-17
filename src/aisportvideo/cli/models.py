import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from ultralytics import RTDETR, YOLO
from ultralytics.engine.model import Model

from aisportvideo.models.rtdetr import (
    RtDetrModelSize,
    download_raw_rtdetr_model_to_path,
)
from aisportvideo.models.ultralytics import exported_model_type_to_ultralytics_format
from aisportvideo.models.yolo import (
    HF_REPOSITORY,
    YoloModelSize,
    YoloModelType,
    get_yolo_hf_filename,
)
from aisportvideo.utils.models import extension_to_model_type

app = typer.Typer()

_logger = logging.getLogger("aisportvideo.cli.models")


def _export_model(model: Model, *, raw_model_path: Path, output_path: Path):
    exported_model_type = extension_to_model_type(output_path.suffix)
    model.export(
        format=exported_model_type_to_ultralytics_format(model_type=exported_model_type),
        dynamic=True,
    )

    exported_model_path = raw_model_path.with_suffix(output_path.suffix)

    if not exported_model_path.is_file():
        err = f"Model not exported to {exported_model_path}"
        raise RuntimeError(err)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported_model_path, output_path)
    if not output_path.is_file():
        err = f"Failed to move model to {output_path}"
        raise RuntimeError(err)


@app.command(name="rtdetr")
def export_rtdetr(
    size: Annotated[RtDetrModelSize, typer.Option(help="Size of the model")],
    output_path: Annotated[Path, typer.Option(help="Path to save the model")],
):
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
        tmp_file_path = Path(tmp_file.name)
        download_raw_rtdetr_model_to_path(tmp_file_path, size=size)

        model = RTDETR(str(tmp_file_path))
        _export_model(model, raw_model_path=tmp_file_path, output_path=output_path)


@app.command(name="yolo11")
def export_yolo11(
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

    model = YOLO(raw_model_path)
    _export_model(model, raw_model_path=raw_model_path, output_path=output_path)
