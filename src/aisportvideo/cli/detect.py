from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()


@app.command(name="object")
def detect_objects(
    path: Annotated[Path, typer.Argument(help="Path to the video file")],
    visualize: Annotated[bool, typer.Option(help="Visualize the detection results")] = False,
):
    import av
    import cv2
    import numpy as np

    from aisportvideo.models.yolo import (
        YoloModelSize,
        YoloModelType,
        load_yolo_from_assets,
        visualize_yolo_results_on_bgr_image,
    )
    from aisportvideo.utils.models import ModelType

    model = load_yolo_from_assets(
        size=YoloModelSize.SMALL, model_type=YoloModelType.DETECTION, export_type=ModelType.ONNX
    )

    container = av.open(path)
    for frame in container.decode(video=0):
        frame_img = frame.to_ndarray(format="bgr24")

        if frame.rotation != 0:
            assert frame.rotation % 90 == 0
            frame_img = np.rot90(frame_img, k=frame.rotation // 90)

        results = model.track(frame_img, persist=True)

        viz_img = (
            visualize_yolo_results_on_bgr_image(results, img=frame_img) if results else frame_img
        )
        if visualize:
            cv2.imshow("test", viz_img)
            cv2.waitKey(0)
