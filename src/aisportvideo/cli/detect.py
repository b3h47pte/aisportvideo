from pathlib import Path
from typing import Annotated

import av
import cv2
import numpy as np
import typer

from aisportvideo.models.huggingface import HuggingFaceSupportedModels
from aisportvideo.models.ultralytics import UltralyticsSupportedModels

app = typer.Typer()


@app.command(name="object_ultra")
def detect_objects_ultralytics(
    path: Annotated[Path, typer.Argument(help="Path to the video file")],
    visualize: Annotated[bool, typer.Option(help="Visualize the detection results")] = False,
    model_type: Annotated[
        UltralyticsSupportedModels, typer.Option(help="Model to use")
    ] = UltralyticsSupportedModels.YOLO,
):
    from aisportvideo.models.rtdetr import RtDetrModelSize, load_rtdetr_from_assets
    from aisportvideo.models.ultralytics import visualize_ultralytics_results_on_bgr_image
    from aisportvideo.models.yolo import (
        YoloModelSize,
        YoloModelType,
        load_yolo_from_assets,
    )
    from aisportvideo.utils.models import ModelType

    match model_type:
        case UltralyticsSupportedModels.YOLO:
            model = load_yolo_from_assets(
                size=YoloModelSize.SMALL,
                model_type=YoloModelType.DETECTION,
                export_type=ModelType.ONNX,
            )
        case UltralyticsSupportedModels.RTDETR:
            model = load_rtdetr_from_assets(size=RtDetrModelSize.LARGE, export_type=ModelType.ONNX)
        case _:
            err = f"Unsupported model type: {model_type}"
            raise ValueError(err)

    container = av.open(path)
    for frame in container.decode(video=0):
        frame_img = frame.to_ndarray(format="bgr24")

        if frame.rotation != 0:
            assert frame.rotation % 90 == 0
            frame_img = np.rot90(frame_img, k=frame.rotation // 90)

        results = model.track(frame_img, persist=True)

        if visualize:
            viz_img = (
                visualize_ultralytics_results_on_bgr_image(results, img=frame_img)
                if results
                else frame_img
            )
            cv2.imshow("test", viz_img)
            cv2.waitKey(0)

@app.command(name="object_hf")
def detect_objects_huggingface(
    path: Annotated[Path, typer.Argument(help="Path to the video file")],
    visualize: Annotated[bool, typer.Option(help="Visualize the detection results")] = False,
    model_type: Annotated[
        HuggingFaceSupportedModels, typer.Option(help="Model to use")
    ] = HuggingFaceSupportedModels.DETR_TENNIS_BALL,
):
    from aisportvideo.models.huggingface import load_hf_pipeline, visualize_hf_results

    pipeline = load_hf_pipeline(model_type)
    container = av.open(path)
    for frame in container.decode(video=0):
        frame_img = frame.to_image()
        results = pipeline(frame_img)

        if visualize:
            viz_img = visualize_hf_results(frame_img, results=results, model=model_type)
            cv2.imshow("test", viz_img[:, :, ::-1])
            cv2.waitKey(0)
