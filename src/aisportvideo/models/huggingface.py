from enum import StrEnum
from typing import Any

import cv2
import numpy as np
from PIL import Image
from transformers import Pipeline, pipeline

from aisportvideo.utils.colors import get_byte_unique_color

_VIZ_FONT_SCALE = 1
_VIZ_FONT_THICKNESS = 2
_MARGIN = 30
_PADDING = 50


class HuggingFaceSupportedModels(StrEnum):
    DETR_TENNIS_BALL = "detr_tennis"
    SEGFORMER_TENNIS = "segformer_tennis"
    DETR_RESNET_50 = "detr_resnet_50"
    DETR_RESNET_101 = "detr_resnet_101"

    @property
    def hf_repo(self) -> str:
        match self:
            case HuggingFaceSupportedModels.DETR_TENNIS_BALL:
                return "SebastianVasquez/detr-finetuned-tennis-ball-v2"
            case HuggingFaceSupportedModels.SEGFORMER_TENNIS:
                return "julia-wenkmann/segformer-b1-finetuned-tennisdata"
            case HuggingFaceSupportedModels.DETR_RESNET_50:
                return "facebook/detr-resnet-50"
            case HuggingFaceSupportedModels.DETR_RESNET_101:
                return "facebook/detr-resnet-101"

    @property
    def pipeline_type(self) -> str:
        match self:
            case (
                HuggingFaceSupportedModels.DETR_TENNIS_BALL
                | HuggingFaceSupportedModels.DETR_RESNET_50
                | HuggingFaceSupportedModels.DETR_RESNET_101
            ):
                return "object-detection"
            case HuggingFaceSupportedModels.SEGFORMER_TENNIS:
                return "image-segmentation"


def load_hf_pipeline(model: HuggingFaceSupportedModels) -> Pipeline:
    return pipeline(model.pipeline_type, model=model.hf_repo)


def _visualize_hf_object_detection_results(img: np.ndarray, *, results: Any) -> np.ndarray:
    for r in results:
        b = r["box"]
        cls_color = get_byte_unique_color(hash(r["label"]))
        top_left = (b["xmin"], b["ymin"])
        bottom_right = (b["xmax"], b["ymax"])

        cv2.rectangle(
            img,
            pt1=top_left,
            pt2=bottom_right,
            color=cls_color,
            thickness=5,
        )

        id_text = f"Class: {r['label']} (Score: {r['score']:.2f}"
        (_, height), _ = cv2.getTextSize(
            id_text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=_VIZ_FONT_SCALE,
            thickness=_VIZ_FONT_THICKNESS,
        )

        cv2.putText(
            img,
            text=id_text,
            org=(top_left[0], top_left[1] - height - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=_VIZ_FONT_SCALE,
            color=cls_color,
            thickness=_VIZ_FONT_THICKNESS,
        )

    return img


def _visualize_hf_image_segmentation_results(img: np.ndarray, *, results: Any) -> np.ndarray:
    def _visualize_single_hf_segmentation(img: np.ndarray, seg: Any) -> np.ndarray:
        mask = np.expand_dims(np.asarray(seg["mask"]), 2)
        cls_color = get_byte_unique_color(hash(seg["label"]))
        viz_mask = np.full_like(img, cls_color)

        return img + 0.3 * viz_mask * mask / 255.0

    for seg in results:
        img = _visualize_single_hf_segmentation(img, seg)

    # Second pass to determine labels and to draw the black background
    max_height = 0
    max_width = 0
    all_text = []
    all_text_dims = []
    for seg in results:
        score = seg["score"] or 0.0
        id_text = f"Class: {seg['label']} (Score: {score:.2f})"
        wh, _ = cv2.getTextSize(
            id_text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=_VIZ_FONT_SCALE,
            thickness=_VIZ_FONT_THICKNESS,
        )

        all_text.append(id_text)
        all_text_dims.append(wh)

        max_width = max(max_width, wh[0])
        max_height += wh[1] + _PADDING

    max_width += 2 * _MARGIN
    max_height += _MARGIN

    cv2.rectangle(img, pt1=(0, 0), pt2=(max_width, max_height), color=(0, 0, 0), thickness=-1)

    # Third pass to actually draw
    for idx, (text, (_, height)) in enumerate(zip(all_text, all_text_dims)):
        cls_color = get_byte_unique_color(hash(results[idx]["label"]))

        score = results[idx]["score"] or 0.0
        cv2.putText(
            img,
            text=text,
            org=(_MARGIN, int(_MARGIN + (height + _PADDING) * idx)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=_VIZ_FONT_SCALE,
            color=cls_color,
            thickness=_VIZ_FONT_THICKNESS,
        )

    return np.clip(img, 0, 255).astype(np.uint8)


def visualize_hf_results(
    img: Image.Image, *, results: Any, model: HuggingFaceSupportedModels
) -> np.ndarray:
    np_img = np.asarray(img).copy()

    match model.pipeline_type:
        case "object-detection":
            return _visualize_hf_object_detection_results(np_img, results=results)
        case "image-segmentation":
            return _visualize_hf_image_segmentation_results(np_img, results=results)
        case _:
            err = f"Unknown pipeline type: {model.pipeline_type}"
            raise ValueError(err)
