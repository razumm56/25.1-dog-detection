import asyncio
import json
import os
import shutil
import tempfile
import threading
import uuid

import cv2
import numpy as np
from cv2.typing import MatLike
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from ultralytics import YOLO

app = FastAPI()

model_path = "./weights/best_30.05.2025.pt"
temp_files_dir = "./temp_frames/"
if not os.path.exists(temp_files_dir):
    os.mkdir(temp_files_dir)
default_conf = 0.25


@app.get("/test")
def test():
    return {"Hello", "FastAPI"}


frame_model = YOLO(model_path)


@app.post("/detect/image")
async def decoded_image(
    frame: UploadFile,
    image_size: int,
    conf: float = default_conf,
):
    nparr = np.frombuffer(await frame.read(), np.uint8)
    decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if decoded_image is None:
        return JSONResponse({"message": "Image is invalid"}, status_code=400)

    response = {
        "detections": process_frame(frame_model, decoded_image, image_size, conf)
    }

    return JSONResponse(content=response)


video_model = YOLO(model_path)


@app.post("/detect/video")
async def video(
    file: UploadFile,
    image_size: int,
    conf: float = default_conf,
    return_annotated: bool = False,
):
    # Сохраняем входное видео во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input_path = temp_input.name
        shutil.copyfileobj(file.file, temp_input)

    cap = cv2.VideoCapture(temp_input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if return_annotated:
        annotated_video_path = f"{temp_input_path}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
    else:
        writer = None

    all_detections = []
    frame_id = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = video_model.predict(frame, imgsz=image_size, conf=conf, verbose=False)
        detections = [r.summary() for r in results]
        all_detections.append({"frame": frame_id, "detections": detections})

        if writer:
            annotated_frame = results[0].plot()
            writer.write(annotated_frame)

        frame_id += 1

    cap.release()
    if writer:
        writer.release()

    os.remove(temp_input.name)

    if return_annotated:
        return FileResponse(
            annotated_video_path,
            media_type="video/mp4",
            filename="annotated_video.mp4",
            headers={"X-Detections-Info": "Full detections in JSON body"},
            background=BackgroundTask(lambda: os.remove(annotated_video_path)),
        )

    return JSONResponse(content={"results": all_detections})


def process_frame(model: YOLO, frame: MatLike, image_size: int, conf: float) -> list:
    results = model.track(
        frame, imgsz=image_size, conf=conf, persist=True, verbose=False
    )

    return [result.summary() for result in results if result.summary()]


def temporary_save_frame(frame: MatLike, dir, timeout: float) -> str:
    frame_id = str(uuid.uuid4())
    filename = f"{dir}/{frame_id}.jpg"
    cv2.imwrite(filename, frame)

    threading.Timer(
        timeout, lambda: os.remove(filename) if os.path.exists(filename) else None
    ).start()

    return frame_id


@app.get("/frame")
def frame(frame_id: str):
    filename = f"{frame_id}.jpg"
    path = os.path.join(temp_files_dir, filename)
    return FileResponse(path, media_type="image/jpeg", filename=filename)


@app.get("/detect/url_stream")
async def url_stream(
    request: Request,
    url: str,
    image_size: int,
    conf: float = default_conf,
):
    model = YOLO("./weights/best_30.05.2025.pt")
    cap = cv2.VideoCapture(url)

    queue = asyncio.Queue(maxsize=1)
    stop_event = asyncio.Event()

    # функция для захвата последнего доступного кадра
    async def capture_frames(
        queue: asyncio.Queue, stop_event: asyncio.Event, cap: cv2.VideoCapture
    ):
        loop = asyncio.get_event_loop()

        while not stop_event.is_set() and cap.isOpened():
            success, frame = await loop.run_in_executor(None, lambda: cap.read())

            if success:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await queue.put(frame)
            await asyncio.sleep(0.01)

    loop = asyncio.get_event_loop()
    grabber_task = loop.create_task(capture_frames(queue, stop_event, cap))

    # функция обработки кадров
    async def gen():
        try:
            frame = None
            while True:
                if await request.is_disconnected():
                    break

                try:
                    frame = await asyncio.wait_for(queue.get(), timeout=1)
                except asyncio.TimeoutError:
                    continue

                data = {}
                detection_results = model.track(
                    frame, imgsz=image_size, conf=conf, persist=True, verbose=False
                )
                data["detections"] = [
                    result.summary() for result in detection_results if result.summary()
                ]
                data["frame_id"] = temporary_save_frame(frame, temp_files_dir, 60)

                yield json.dumps(data) + "\n"

        finally:
            stop_event.set()
            cap.release()
            grabber_task.cancel()

    return StreamingResponse(gen(), media_type="text/event-stream")
