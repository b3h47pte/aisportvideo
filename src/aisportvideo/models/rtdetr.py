import importlib.resources
from enum import StrEnum
from pathlib import Path

import httpx
from ultralytics import RTDETR

from aisportvideo.utils.models import ModelType


class RtDetrModelSize(StrEnum):
    LARGE = "large"
    XL = "xlarge"

    @property
    def rtdetr_filename(self) -> str:
        return f"rtdetr-{self.value[0]}.pt"

    @property
    def rtdetr_url(self) -> str:
        return (
            f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{self.rtdetr_filename}"
        )


def download_raw_rtdetr_model_to_path(path: Path, *, size: RtDetrModelSize):
    url = size.rtdetr_url
    r = httpx.get(url, follow_redirects=True)
    with path.open("wb") as f:
        f.write(r.content)
    assert path.is_file()

def load_rtdetr_from_assets(
    *, size: RtDetrModelSize, export_type: ModelType
) -> RTDETR:
    with importlib.resources.path(
        "assets", size.rtdetr_filename
    ) as asset_path:
        final_path = asset_path.with_suffix(export_type.extension)
        assert final_path.exists()
        return RTDETR(str(final_path))
