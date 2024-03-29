# -*- coding: utf-8 -*-

# @Time    : 2024/3/28 17:19
# @Author  : kewei

from typing import Union, Tuple, Dict, Any, List
from pydantic import BaseModel
import time
from collections import defaultdict


class OcrKwargs(BaseModel):
    rec_batch_size: int = 100
    return_cropped_image: bool = False
    resized_shape: Union[int, Tuple[int, int]] = (768, 768)
    preserve_aspect_ratio: bool = True
    min_box_size: int = 8
    box_score_thresh: float = 0.3
    batch_size: int = 20


class OcrResponse(BaseModel):
    results: list
    message: str = 'success'
    code: int = 0


class OcrBatchResponse(BaseModel):
    results: dict
    message: str = 'success'
    code: int = 0


class QueueRequest(BaseModel):
    task_id: str
    image: Any
    ocr_kwargs: OcrKwargs = None


class Response(BaseModel):
    task_id: str
    response: list
    create_time: float = time.time()


class QueueResponse(BaseModel):
    task_map: Dict[str, List[Response]] = defaultdict(list)
    status: int = 0
