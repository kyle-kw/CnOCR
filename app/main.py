# -*- coding: utf-8 -*-

# @Time    : 2024/3/28 16:56
# @Author  : kewei

import time
from typing import List
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.exceptions import HTTPException

from app.utils import generate_task_id, put_task, get_response, check_file, logger, \
    start_daemon_thread
from app.models import OcrKwargs, OcrResponse, OcrBatchResponse, QueueRequest, Response

start_daemon_thread()
app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to OCR Server!"}


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/v1/batch/ocr")
def ocr(image_lst: List[UploadFile] = File(...),
        ocr_kwargs: OcrKwargs = Form(default=OcrKwargs()),
        timeout: int = Form(default=10)
        ) -> OcrBatchResponse:
    logger.debug(f"batch/ocr: Received {len(image_lst)} images")

    task_id_lst = dict()
    for image in image_lst:
        if image.content_type not in ('image/jpeg', 'image/png'):
            raise HTTPException(status_code=400, detail="Unsupported file type")

        image_obj = check_file(image)
        task_id = generate_task_id()
        put_task(QueueRequest(task_id=task_id, image=image_obj, ocr_kwargs=ocr_kwargs),
                 timeout=timeout)
        task_id_lst[task_id] = image.filename
    logger.debug(f"batch/ocr: Task IDs: {task_id_lst}")

    unfinished = True
    res = list()
    now = time.time()
    while unfinished and (time.time() - now) < timeout:
        time.sleep(0.1)
        unfinished = False
        for task_id, file_name in task_id_lst.items():
            response_data: List[Response] = get_response(task_id)
            if response_data:
                res.append({
                    "image": file_name or "",
                    "result": response_data[0].response
                })
            else:
                unfinished = True

    logger.debug(f"batch/ocr: Results: {res}")
    if not unfinished:
        logger.info(f"batch/ocr: Finished {len(image_lst)} images")
    else:
        logger.warning(f"batch/ocr: Failed to finish {len(image_lst)} images within {timeout} seconds")

    return OcrBatchResponse(results=res)


@app.post("/v1/ocr")
def one_ocr(image: UploadFile = File(...),
            ocr_kwargs: OcrKwargs = Form(default=OcrKwargs()),
            timeout: int = Form(default=10)
            ) -> OcrResponse:
    logger.debug("ocr: Received image")

    image_obj = check_file(image)
    task_id = generate_task_id()
    put_task(QueueRequest(task_id=task_id, image=image_obj, ocr_kwargs=ocr_kwargs),
             timeout=timeout)
    logger.debug(f"ocr: Task ID: {task_id}")

    unfinished = True
    res = None
    now = time.time()
    while unfinished and (time.time() - now) < timeout:
        time.sleep(0.1)
        response_data: List[Response] = get_response(task_id)
        if response_data:
            res = [d.response for d in response_data]
            unfinished = False

    logger.debug(f"ocr: Results: {res}")
    if not unfinished:
        logger.info("ocr: Finished 1 image")
    else:
        logger.warning(f"ocr: Failed to finish 1 image within {timeout} seconds")

    return OcrResponse(results=res)
