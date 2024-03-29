# -*- coding: utf-8 -*-

# @Time    : 2024/3/28 16:56
# @Author  : kewei

import time
from typing import List
from fastapi import FastAPI, UploadFile, Form, File
# from fastapi.responses import JSONResponse

from app.utils import generate_task_id, put_task, get_response, check_file, logger, \
    start_daemon_thread
from app.models import OcrKwargs, OcrResponse, OcrBatchResponse, QueueRequest, Response

start_daemon_thread()
app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to OCR Server!"}


@app.get("/ping")
def root():
    return {"message": "pong"}


@app.post("/batch/ocr")
def ocr(image_lst: List[UploadFile] = File(...),
        ocr_kwargs: OcrKwargs = Form(default=OcrKwargs()),
        timeout: int = Form(default=10)
        ) -> OcrBatchResponse:
    logger.debug(f"batch/ocr: Received {len(image_lst)} images")

    task_id_lst = []
    for image in image_lst:
        image_obj = check_file(image)
        task_id = generate_task_id()
        put_task(QueueRequest(task_id=task_id, image=image_obj, ocr_kwargs=ocr_kwargs),
                 timeout=timeout)
        task_id_lst.append(task_id)
    logger.debug(f"batch/ocr: Task IDs: {task_id_lst}")

    unfinished = True
    res = dict()
    now = time.time()
    while unfinished and (time.time() - now) < timeout:
        time.sleep(0.1)
        unfinished = False
        for i, task_id in enumerate(task_id_lst):
            response_data: List[Response] = get_response(task_id)
            if response_data:
                res[i] = [d.response for d in response_data]
            else:
                unfinished = True

    logger.debug(f"batch/ocr: Results: {res}")
    if not unfinished:
        logger.info(f"batch/ocr: Finished {len(image_lst)} images")
    else:
        logger.warning(f"batch/ocr: Failed to finish {len(image_lst)} images within {timeout} seconds")

    return OcrBatchResponse(results=res)


@app.post("/ocr")
def ocr(image: UploadFile = File(...),
        ocr_kwargs: OcrKwargs = Form(default=OcrKwargs()),
        timeout: int = Form(default=10)
        ) -> OcrResponse:
    logger.debug(f"ocr: Received image")

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
        logger.info(f"ocr: Finished 1 image")
    else:
        logger.warning(f"ocr: Failed to finish 1 image within {timeout} seconds")

    return OcrResponse(results=res)
