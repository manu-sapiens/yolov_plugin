# --------------- SHARED ---------------------------------------------------
import sys
from typing import List, Any
sys.path.append('.')  # Add the current directory to the sys path
sys.path.append('utils')  # Add the utils directory to the sys path

from utils.omni_utils_http import CdnResponse, ImageMeta, create_api_route, plugin_main, init_plugin
from pydantic import BaseModel
app, router = init_plugin()
# ---------------------------------------------------------------------------
plugin_module_name="Plugins.yolov_plugin.yolov"

# --------------- ENDPOINT_YOLOV_SEGMENT -----------------------------
ENDPOINT_YOLOV_SEGMENT = "/yolov/segment"

class YolovSegment_Input(BaseModel):

    images: List[CdnResponse]
    categories_csv: str
    model: str
    output_mask: bool
    output_merged: bool
    output_alpha: bool
    invert_mask: bool
    minimum_confidence: float

    class Config:
        schema_extra = {
            "title": "Yolov: Segmentation"
        }

class YolovSegment_Response(BaseModel):
    passthrough_array: List[CdnResponse]
    media_array: List[CdnResponse]    
    json_array: List[Any]

    class Config:
        schema_extra = {
            "title": "Yolov: Segmentation"
        }

YolovSegment_Post = create_api_route(
    app=app,
    router=router,
    context=__name__,
    endpoint=ENDPOINT_YOLOV_SEGMENT,
    input_class=YolovSegment_Input,
    response_class=YolovSegment_Response,
    handle_post_function="integration_YolovSegment_Post",
    plugin_module_name=plugin_module_name
)

endpoints = [ENDPOINT_YOLOV_SEGMENT]

# --------------- SHARED ---------------------------------------------------
plugin_main(app, __name__, __file__)
# --------------------------------------------------------------------------