import importlib.resources
from enum import StrEnum
from typing import cast

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from aisportvideo.utils.colors import get_byte_unique_color
from aisportvideo.utils.models import ModelType

HF_REPOSITORY: str = "Ultralytics/YOLO11"


class YoloModelSize(StrEnum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xlarge"

    @property
    def yolo_filename(self) -> str:
        return self.value[0]


class YoloModelType(StrEnum):
    DETECTION = "detect"
    SEGMENTATION = "segment"
    POSE = "pose"

    @property
    def yolo_filename(self) -> str:
        match self:
            case YoloModelType.DETECTION:
                return ""
            case YoloModelType.SEGMENTATION:
                return "-seg"
            case YoloModelType.POSE:
                return "-pose"
            case _:
                err = f"Unsupported model type: {self}"
                raise ValueError(err)


def get_yolo_hf_filename(*, size: YoloModelSize, model_type: YoloModelType) -> str:
    return f"yolo11{size.yolo_filename}{model_type.yolo_filename}.pt"


def exported_model_type_to_yolo_format(model_type: ModelType) -> str:
    match model_type:
        case ModelType.ONNX:
            return "onnx"
        case ModelType.COREML_PACKAGE:
            return "coreml"
        case _:
            err = f"Unsupported model type: {model_type}"
            raise ValueError(err)


def load_yolo_from_assets(
    *, size: YoloModelSize, model_type: YoloModelType, export_type: ModelType
) -> YOLO:
    with importlib.resources.path(
        "assets", get_yolo_hf_filename(size=size, model_type=model_type)
    ) as asset_path:
        final_path = asset_path.with_suffix(export_type.extension)
        assert final_path.exists()
        return YOLO(final_path, task=model_type.value)


_VIZ_FONT_SCALE = 1
_VIZ_FONT_THICKNESS = 2


def _draw_yolo_boxes_on_bgr_image(boxes: Boxes, *, img: np.ndarray, class_mapping: dict[int, str]):
    for b in boxes:
        assert isinstance(b.xyxy, torch.Tensor)
        assert isinstance(b.cls, torch.Tensor)
        assert b.id is None or isinstance(b.id, torch.Tensor)

        top_left, bottom_right = torch.split(b.xyxy[0].int(), 2)
        id: int = cast(int, b.id.int().item()) if b.id is not None else -1
        cls: int = cast(int, b.cls.int().item())

        cls_color = cast(cv2.typing.Scalar, get_byte_unique_color(cast(int, cls)))
        cv2.rectangle(
            img,
            pt1=top_left.tolist(),
            pt2=bottom_right.tolist(),
            color=cls_color,
            thickness=5,
        )

        id_text = f"Class: {class_mapping[cls]} (ID: {id})"
        (_, height), _ = cv2.getTextSize(
            id_text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=_VIZ_FONT_SCALE,
            thickness=_VIZ_FONT_THICKNESS,
        )

        cv2.putText(
            img,
            text=id_text,
            org=(cast(int, top_left[0].item()), cast(int, top_left[1].item()) - height),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=_VIZ_FONT_SCALE,
            color=cls_color,
            thickness=_VIZ_FONT_THICKNESS,
        )


def visualize_yolo_results_on_bgr_image(results: list[Results], *, img: np.ndarray) -> np.ndarray:
    viz_img = img.copy()

    for r in results:
        if r.boxes is None:
            continue
        _draw_yolo_boxes_on_bgr_image(r.boxes, img=viz_img, class_mapping=r.names)

    return viz_img
