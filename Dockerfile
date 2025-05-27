FROM python:3.13

WORKDIR /app

COPY ./pyproject.toml ./requirements.txt ./

RUN pip install --upgrade pip && pip install -r ./requirements.txt

COPY ./src/ ./src/

ENV PYTHONPATH=/app/src

CMD ["python", "./src/detection_cycle.py"]
