from enum import StrEnum

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
