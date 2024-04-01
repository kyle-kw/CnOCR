# -*- coding: utf-8 -*-

# @Time    : 2024/3/28 16:57
# @Author  : kewei

from pathlib import Path
from typing import Optional, Union, Collection, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
from cnocr.utils import data_dir
from cnstd.utils import data_dir as det_data_dir
from enum import Enum


class Context(str, Enum):
    cpu = 'cpu'
    gpu = 'gpu'
    cuda = 'cuda'


class ModelBackend(str, Enum):
    pytorch = 'pytorch'
    onnx = 'onnx'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    rec_model_name: str = 'densenet_lite_136-gru'
    det_model_name: str = 'ch_PP-OCRv3_det'
    cand_alphabet: Optional[Union[Collection, str]] = None
    context: Context = 'cpu'
    rec_model_fp: Optional[str] = None
    rec_model_backend: ModelBackend = 'onnx'
    rec_vocab_fp: Optional[Union[str, Path]] = None
    rec_more_configs: Optional[Dict[str, Any]] = None
    rec_root: Union[str, Path] = data_dir()
    det_model_fp: Optional[str] = None
    det_model_backend: ModelBackend = 'onnx'
    det_more_configs: Optional[Dict[str, Any]] = None
    det_root: Union[str, Path] = det_data_dir()


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    queue_maxsize: int = 100
    queue_timeout: float = 5.0
    queue_return: int = 10

    response_clear_time: int = 600
    response_clear_sleep: int = 10


settings = json.loads(Settings().json(exclude_defaults=True))
env_settings = EnvSettings()

if __name__ == '__main__':
    # import os
    # os.environ['CONTEXT'] = 'gpu'
    print(settings)
    print(env_settings)
