
FROM ccr.ccs.tencentyun.com/zhongbiao/onnx-env:v2

RUN pip install loguru pymupdf pydantic-settings

ENV PYTHONPATH=/app

WORKDIR /app
COPY . /app/app

RUN python3 -c "from app.utils import init_cnocr;print(init_cnocr())"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
