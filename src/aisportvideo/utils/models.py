from enum import StrEnum


class ModelType(StrEnum):
    ONNX = "onnx"
    COREML_PACKAGE = "mlpackage"

_EXTENSION_MAPPING = {
    model_type.value: model_type
    for model_type in ModelType
}

def extension_to_model_type(extension: str) -> ModelType:
    return _EXTENSION_MAPPING[extension.strip(".").lower()]