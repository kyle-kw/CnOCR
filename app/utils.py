# -*- coding: utf-8 -*-

# @Time    : 2024/3/28 17:10
# @Author  : kewei

import sys
import fitz
import torch
import queue
import time
import random
import threading
import numpy as np
from PIL import Image, ImageSequence
from pathlib import Path
from fastapi import UploadFile
from fastapi.exceptions import HTTPException
from typing import Union, List, Dict, Any
from string import digits, ascii_lowercase, ascii_uppercase
from loguru import logger

from cnocr import CnOcr
from cnocr.cn_ocr import OcrResult
from cnocr.line_split import line_split
from app.models import QueueResponse, QueueRequest, Response
from app.config import settings, env_settings

logger.remove(0)
logger.add("logs/info.log", rotation="10 MB", compression="zip")
logger.add(sys.stdout, level="INFO")

logger.info(settings)
logger.info(env_settings)

q = queue.Queue(maxsize=env_settings.queue_maxsize)
q_response = QueueResponse()
q_lock = threading.Lock()


class BatchCnOcr(CnOcr):

    def _deal_and_split_image(self,
                              img_fp: Union[str, Path, Image.Image, torch.Tensor, np.ndarray]
                              ) -> List[Image.Image]:
        if isinstance(img_fp, Image.Image):  # Image to np.ndarray
            img_fp = np.asarray(img_fp.convert('RGB'))

        img = self._prepare_img(img_fp)

        if min(img.shape[0], img.shape[1]) < 2:
            return []
        if img.mean() < 145:  # 把黑底白字的图片对调为白底黑字
            img = 255 - img
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=-1)
        line_images = line_split(img, blank=True)

        return line_images

    def batch_ocr(self,
                  img_fp_lst: List[Union[str, Path, Image.Image, torch.Tensor, np.ndarray]],
                  rec_batch_size=100,
                  return_cropped_image=False,
                  **det_kwargs,
                  ) -> Dict[int, List[Dict[str, Any]]]:
        logger.debug(f"batch_ocr: Received {len(img_fp_lst)} images")

        if self.det_model is not None:
            return self._batch_ocr_with_det_model(
                img_fp_lst, rec_batch_size, return_cropped_image, **det_kwargs
            )

        img_map = {}
        line_img_all = []
        for i, img_fp in enumerate(img_fp_lst):
            line_images = self._deal_and_split_image(img_fp)
            img_map[i] = len(line_images)
            line_img_all.extend(line_images)

        line_img_list = [line_img for line_img, _ in line_img_all]
        line_chars_list = self.ocr_for_single_lines(
            line_img_list, batch_size=rec_batch_size
        )
        if return_cropped_image:
            for _out, line_img in zip(line_chars_list, line_img_list):
                _out['cropped_img'] = line_img

        res_chars_map = {}
        idx = 0
        for i, num in img_map.items():
            res_chars_map[i] = line_chars_list[idx: idx + num]
            idx += num
        logger.debug(f"batch_ocr: Results: {res_chars_map}")

        return res_chars_map

    def _batch_ocr_with_det_model(self,
                                  img_lst: List[Union[str, Path, torch.Tensor, np.ndarray, Image.Image]],
                                  rec_batch_size: int,
                                  return_cropped_image: bool,
                                  **det_kwargs,
                                  ) -> Dict[int, List[Dict[str, Any]]]:
        for i, img in enumerate(img_lst):
            if isinstance(img, Image.Image):  # Image to np.ndarray
                img_lst[i] = np.asarray(img.convert('RGB'))

            if isinstance(img, torch.Tensor):
                img_lst[i] = img.numpy()
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3 and img.shape[2] == 1:
                    # (H, W, 1) -> (H, W)
                    img_lst[i] = img.squeeze(-1)
                if len(img.shape) == 2:
                    # (H, W) -> (H, W, 3)
                    img_lst[i] = np.array(Image.fromarray(img).convert('RGB'))

        box_infos = self.det_model.detect(img_lst, **det_kwargs)

        img_map = {}
        cropped_img_list = []
        det_text_list = []
        for i, one_box_infos in enumerate(box_infos):
            for box_info in one_box_infos['detected_texts']:
                cropped_img_list.append(box_info['cropped_img'])

            det_text_list.extend(one_box_infos['detected_texts'])
            img_map[i] = len(one_box_infos['detected_texts'])

        ocr_outs = self.ocr_for_single_lines(
            cropped_img_list, batch_size=rec_batch_size
        )
        results = []
        for box_info, ocr_out in zip(det_text_list, ocr_outs):
            _out = OcrResult(**ocr_out)
            _out.position = box_info['box']
            if return_cropped_image:
                _out.cropped_img = box_info['cropped_img']
            results.append(_out.to_dict())

        results_map = {}
        idx = 0
        for i, num in img_map.items():
            results_map[i] = results[idx: idx + num]
            idx += num
        logger.debug(f"batch_ocr_with_det_model: Results: {results_map}")
        return results_map


def get_task_lst(timeout: float = env_settings.queue_timeout,
                 number: int = env_settings.queue_return
                 ) -> List[QueueRequest]:
    now = time.time()
    task_lst = []
    delay_time = 0
    while delay_time < timeout and len(task_lst) < number:
        try:
            task = q.get(timeout=timeout - delay_time)
            task_lst.append(task)
        except queue.Empty:
            pass
        delay_time = time.time() - now

    logger.debug(f"get_task_lst: Received {len(task_lst)} tasks")
    return task_lst


def deal_ocr_task():
    cnocr = BatchCnOcr(**settings)
    while True:
        try:
            logger.debug("deal_ocr_task: Start")
            task_lst: List[QueueRequest] = get_task_lst()
            if not task_lst:
                continue
            task_id_lst = []
            img_lst = []
            for task in task_lst:
                task_id = task.task_id
                for img in task.image:
                    img_lst.append(img)
                    task_id_lst.append(task_id)
            res = cnocr.batch_ocr(img_lst)
            with q_lock:
                for task_id, one_res in zip(task_id_lst, res.values()):
                    for _one in one_res:
                        _one['position'] = _one['position'].tolist()
                        _one['score'] = float(_one['score'])
                        if 'cropped_img' in _one:
                            _one.pop('cropped_img')
                    q_response.task_map[task_id].append(
                        Response(
                            task_id=task_id,
                            response=one_res,
                            create_time=time.time()
                        ))
            logger.debug("deal_ocr_task: Finished")
        except Exception as e:
            logger.exception(e)
            cnocr = BatchCnOcr(**settings)


def deal_expired_response():
    while True:
        try:
            logger.debug("deal_expired_response: Start")

            with q_lock:
                now = time.time()

                expire_id = []
                for task_id, response_lst in q_response.task_map.items():
                    for response in response_lst:

                        if (now - response.create_time) > env_settings.response_clear_time:
                            expire_id.append(task_id)
                            break

                for task_id in expire_id:
                    q_response.task_map.pop(task_id, None)
                    logger.warning(f"deal_expired_response: Task {task_id} expired")

            logger.debug("deal_expired_response: Finished")
        except Exception as e:
            logger.exception(e)

        time.sleep(env_settings.response_clear_sleep)


def start_daemon_thread():
    deal_ocr_thread = threading.Thread(target=deal_ocr_task)
    deal_ocr_thread.daemon = True
    deal_ocr_thread.start()

    deal_expired_thread = threading.Thread(target=deal_expired_response)
    deal_expired_thread.daemon = True
    deal_expired_thread.start()


def generate_task_id(task_len=16) -> str:
    return ''.join(random.choices(digits + ascii_lowercase + ascii_uppercase, k=task_len))


def put_task(task: QueueRequest, timeout=3):
    try:
        q.put(task, timeout=timeout)
    except queue.Full:
        logger.warning("put_task: Queue is full")
        raise HTTPException(status_code=503, detail="Service Unavailable")

    return task.task_id


def get_response(task_id: str) -> List[Response]:
    with q_lock:
        return q_response.task_map.pop(task_id, None)


def check_file(file: UploadFile) -> List[Image.Image]:
    if file.content_type == 'image/gif':
        images = []
        for frame in ImageSequence.Iterator(Image.open(file.file)):
            frame = frame.convert('RGB')
            images.append(frame)

        return images

    elif file.content_type in ('image/jpeg', 'image/png'):
        image_obj = Image.open(file.file).convert('RGB')
        return [image_obj]

    elif file.content_type == 'application/pdf':
        images = []
        with fitz.open(stream=file.file.read(), filetype="pdf") as pdf:
            for page in pdf:
                pm = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
                images.append(img)

        return images

    raise HTTPException(status_code=400, detail="Unsupported file type")


def init_cnocr():
    cnocr = BatchCnOcr(**settings)
    return cnocr
